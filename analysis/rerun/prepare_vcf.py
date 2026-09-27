"""Build a strict, phased SNP NPY cache from an original cohort VCF.

VCF POS is retained as exact one-based int64, matching inclusive GENCODE gene
coordinates in the empirical analyses. All rows at repeated positions are
removed, including rows that would individually fail the SNP/FILTER criteria.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chromosome_name(value):
    value = str(value)
    return value[3:] if value.lower().startswith("chr") else value


def unique_records(records, chromosome, counts):
    """Yield only singleton coordinate groups; reject unsorted/mixed input."""
    pending = None
    previous = None
    group_size = 0
    for record in records:
        pos = int(record.POS)
        if chromosome_name(record.CHROM) != chromosome_name(chromosome):
            raise ValueError(f"Unexpected chromosome {record.CHROM}; expected {chromosome}")
        if pos < 1 or (previous is not None and pos < previous):
            raise ValueError(f"VCF positions must be positive and nondecreasing: {pos} after {previous}")
        counts["input_records"] += 1
        if previous is not None and pos != previous:
            if group_size == 1:
                counts["unique_positions"] += 1
                yield pending
            else:
                counts["duplicate_position_groups"] += 1
                counts["duplicate_position_records"] += group_size
            group_size = 0
        pending = record
        previous = pos
        group_size += 1
    if group_size == 1:
        counts["unique_positions"] += 1
        yield pending
    elif group_size:
        counts["duplicate_position_groups"] += 1
        counts["duplicate_position_records"] += group_size


def retained_haplotypes(record, n_samples, counts):
    """Return diploid REF/ALT haplotypes or count one exclusive rejection."""
    if record.FILTER not in (None, "PASS", "."):
        counts["rejected_filter"] += 1
        return None
    alt = record.ALT
    if (record.REF not in ("A", "C", "G", "T") or len(alt) != 1
            or alt[0] not in ("A", "C", "G", "T") or alt[0] == record.REF):
        counts["rejected_not_biallelic_acgt_snp"] += 1
        return None
    # cyvcf2 returns one allele column per ploidy, then a phased-status column.
    # The copy also separates the returned values from cyvcf2 record storage.
    gt = np.asarray(record.genotype.array())
    if gt.shape != (n_samples, 3):
        counts["rejected_non_diploid"] += 1
        return None
    alleles = gt[:, :2]
    if np.any((alleles != 0) & (alleles != 1)):
        counts["rejected_incomplete_or_nonbinary_genotype"] += 1
        return None
    if np.any((alleles[:, 0] != alleles[:, 1]) & (gt[:, 2] == 0)):
        counts["rejected_unphased_heterozygote"] += 1
        return None
    counts["retained_sites"] += 1
    counts["retained_unphased_homozygous_genotypes"] += int(np.count_nonzero(gt[:, 2] == 0))
    return alleles.astype(np.uint8, copy=True).reshape(-1)


def open_vcf(path, threads):
    from cyvcf2 import VCF
    return VCF(str(path), threads=threads)


def scan(path, chromosome, samples, threads, consume=None):
    reader = open_vcf(path, threads)
    try:
        if list(reader.samples) != samples:
            raise ValueError("VCF sample IDs/order changed between passes")
        counts = Counter({key: 0 for key in (
            "input_records", "unique_positions", "duplicate_position_groups",
            "duplicate_position_records", "rejected_filter", "rejected_not_biallelic_acgt_snp",
            "rejected_non_diploid", "rejected_incomplete_or_nonbinary_genotype",
            "rejected_unphased_heterozygote", "retained_sites",
            "retained_unphased_homozygous_genotypes")})
        for record in unique_records(reader, chromosome, counts):
            haplotypes = retained_haplotypes(record, len(samples), counts)
            if haplotypes is not None and consume is not None:
                consume(int(record.POS), haplotypes)
        return dict(sorted(counts.items()))
    finally:
        reader.close()


def source_state(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare(source, output, chromosome, *, expected_samples=3202, chunk_sites=8192,
            threads=1, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunk_sites < 1 or threads < 1 or (expected_samples is not None and expected_samples < 1):
        raise ValueError("Chunk size, thread count and expected sample count must be positive")
    initial_state = source_state(source)
    source_hash = sha256(source)
    config = {"chromosome": chromosome_name(chromosome), "expected_samples": expected_samples,
              "coordinate_convention": "VCF POS: one-based integer; gene intervals one-based inclusive",
              "positions_base": 1,
              "variant_filter": "PASS or dot; biallelic distinct A/C/G/T SNP; complete diploid genotypes; phased heterozygotes",
              "duplicate_rule": "remove every row at any physical position appearing more than once, before other filters",
              "haplotype_order": "VCF sample order, allele 0 then allele 1 for each sample"}
    identity = {"source_sha256": source_hash, "configuration": config,
                "preparer_sha256": sha256(__file__)}
    if output.exists():
        marker = output / "READY.json"
        if resume and marker.is_file():
            previous = json.loads(marker.read_text())
            if previous.get("identity") == identity and all(
                    (output / name).is_file() and (output / name).stat().st_size == member["bytes"]
                    and sha256(output / name) == member["sha256"]
                    for name, member in previous.get("members", {}).items()) and len(previous.get("members", {})) == 3:
                print(json.dumps({"status": "already_complete", "output": str(output)}), flush=True)
                return previous
        raise FileExistsError(f"Refusing to replace existing/incomplete/mismatched cache: {output}")
    reader = open_vcf(source, threads)
    try:
        samples = list(reader.samples)
        if not samples or len(samples) != len(set(samples)):
            raise ValueError("VCF must contain unique nonempty sample IDs")
        if expected_samples is not None and len(samples) != expected_samples:
            raise ValueError(f"VCF has {len(samples)} samples; expected {expected_samples}")
        sequence_length = None
        try:
            lengths = reader.seqlens
            for name, length in zip(reader.seqnames, lengths):
                if chromosome_name(name) == chromosome_name(chromosome) and length:
                    sequence_length = int(length)
        except (AttributeError, TypeError, ValueError):
            pass  # A contig-length header is optional; POS remains fully specified.
    finally:
        reader.close()
    first_counts = scan(source, chromosome, samples, threads)
    n_sites = first_counts.get("retained_sites", 0)
    if not n_sites:
        raise ValueError(f"No sites retained: {first_counts}")
    if source_state(source) != initial_state:
        raise RuntimeError("Source VCF changed during the first pass")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        G = np.lib.format.open_memmap(temp / "G.npy", mode="w+", dtype=np.uint8,
                                     shape=(2 * len(samples), n_sites))
        positions = np.lib.format.open_memmap(temp / "positions.npy", mode="w+", dtype=np.int64,
                                             shape=(n_sites,))
        buffer = np.empty((min(chunk_sites, n_sites), 2 * len(samples)), dtype=np.uint8)
        pos_buffer = np.empty(len(buffer), dtype=np.int64)
        cursor, buffered = 0, 0

        def flush():
            nonlocal cursor, buffered
            if buffered:
                if cursor + buffered > n_sites:
                    raise RuntimeError("Second pass retained more sites than the first")
                G[:, cursor:cursor + buffered] = buffer[:buffered].T
                positions[cursor:cursor + buffered] = pos_buffer[:buffered]
                cursor += buffered
                buffered = 0

        def consume(position, haplotypes):
            nonlocal buffered
            buffer[buffered] = haplotypes
            pos_buffer[buffered] = position
            buffered += 1
            if buffered == len(buffer):
                flush()

        second_counts = scan(source, chromosome, samples, threads, consume)
        flush()
        if second_counts != first_counts or cursor != n_sites:
            raise RuntimeError("VCF filters/counts changed between passes")
        if source_state(source) != initial_state:
            raise RuntimeError("Source VCF changed during conversion")
        if np.any(np.diff(positions) <= 0) or positions[0] < 1:
            raise RuntimeError("Internal error: retained positions are not strictly increasing")
        first_position, last_position = int(positions[0]), int(positions[-1])
        if sequence_length is not None and last_position > sequence_length:
            raise ValueError("Retained POS exceeds the VCF contig length")
        G.flush(); positions.flush()
        del G, positions, buffer, pos_buffer
        np.save(temp / "sample_ids.npy", np.asarray(samples, dtype=str), allow_pickle=False)
        members = {name: {"sha256": sha256(temp / name), "bytes": (temp / name).stat().st_size}
                   for name in ("G.npy", "positions.npy", "sample_ids.npy")}
        record = {"schema": 2, "identity": identity, "source_name": source.name,
                  "source_path": str(source), "source_bytes": initial_state["bytes"],
                  "source_sha256": source_hash, "members": members,
                  "shape": [2 * len(samples), n_sites], "dtype": "uint8", "positions_dtype": "int64",
                  "samples": len(samples), "first_position": first_position, "last_position": last_position,
                  "coordinate_convention": config["coordinate_convention"], "positions_base": 1,
                  "vcf_contig_length_bp": sequence_length,
                  "historical_calibration_denominator": "last retained VCF POS + 1 unless an explicit sequence length is supplied",
                  "filters": config, "filter_counts": first_counts,
                  "execution": {"chunk_sites": chunk_sites, "threads": threads, "passes": 2},
                  "versions": {"python": sys.version, "numpy": np.__version__,
                               "cyvcf2": importlib.metadata.version("cyvcf2")}}
        (temp / "READY.json").write_text(json.dumps(record, indent=2) + "\n")
        os.rename(temp, output)
        print(json.dumps(record), flush=True)
        return record
    except BaseException:
        shutil.rmtree(temp)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--output", required=True, help="New cache chromosome directory, e.g. INPUTS/cache/parsed/chr22")
    parser.add_argument("--chr", required=True, dest="chromosome")
    parser.add_argument("--expected-samples", type=int, default=3202)
    parser.add_argument("--chunk-sites", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare(args.vcf, args.output, args.chromosome, expected_samples=args.expected_samples,
            chunk_sites=args.chunk_sites, threads=args.threads, resume=args.resume)


if __name__ == "__main__":
    main()
