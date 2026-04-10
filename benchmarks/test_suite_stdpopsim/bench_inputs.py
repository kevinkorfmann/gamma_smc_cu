import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedInputs:
    G: np.ndarray
    pos: np.ndarray
    n_total_records: int
    n_kept_records: int
    n_dropped_non_snp: int
    n_dropped_non_binary: int
    n_dropped_missing: int
    n_dropped_nonseg: int


def _parse_gt_field(sample_field: str) -> list[int] | None:
    token = sample_field.split(":", 1)[0]
    if token in {".", "./.", ".|."}:
        return None
    if "|" in token:
        parts = token.split("|")
    elif "/" in token:
        parts = token.split("/")
    else:
        return None
    if len(parts) != 2:
        return None
    try:
        alleles = [int(part) for part in parts]
    except ValueError:
        return None
    if any(allele < 0 for allele in alleles):
        return None
    return alleles


def materialize_binary_snp_vcf(ts, out_vcf_path: str) -> PreparedInputs:
    """Write a tmrca.cu-compatible VCF and return the matching inputs.

    gamma_smc reads HTSlib's 0-based positions and skips non-SNP records.
    tmrca.cu currently expects a binary haplotype matrix, so we keep only
    biallelic SNP records with diploid genotypes in {0, 1} and no missing data.
    """
    os.makedirs(os.path.dirname(out_vcf_path), exist_ok=True)
    full_vcf_path = out_vcf_path + ".full"
    with open(full_vcf_path, "w") as f:
        ts.write_vcf(f, contig_id="chr1", allow_position_zero=True)

    positions = []
    site_haplotypes = []
    n_total_records = 0
    n_dropped_non_snp = 0
    n_dropped_non_binary = 0
    n_dropped_missing = 0
    n_dropped_nonseg = 0

    with open(full_vcf_path) as src, open(out_vcf_path, "w") as dst:
        for line in src:
            if line.startswith("#"):
                dst.write(line)
                continue

            n_total_records += 1
            fields = line.rstrip("\n").split("\t")
            ref = fields[3]
            alt = fields[4].split(",")
            if len(ref) != 1 or any(len(a) != 1 or a == "." for a in alt):
                n_dropped_non_snp += 1
                continue

            haplotypes = []
            has_alt = False
            bad_record = False
            missing_record = False

            for sample_field in fields[9:]:
                gt = _parse_gt_field(sample_field)
                if gt is None:
                    missing_record = True
                    bad_record = True
                    break
                if any(allele not in (0, 1) for allele in gt):
                    bad_record = True
                    break
                haplotypes.extend(gt)
                has_alt = has_alt or any(allele == 1 for allele in gt)

            if bad_record:
                if missing_record:
                    n_dropped_missing += 1
                else:
                    n_dropped_non_binary += 1
                continue

            if not has_alt:
                n_dropped_nonseg += 1
                continue

            dst.write(line)
            positions.append(float(int(fields[1]) - 1))
            site_haplotypes.append(haplotypes)

    os.remove(full_vcf_path)

    if not site_haplotypes:
        raise RuntimeError("no binary SNP records remained after VCF normalization")

    G = np.asarray(site_haplotypes, dtype=np.uint8).T
    pos = np.asarray(positions, dtype=np.float64)
    return PreparedInputs(
        G=G,
        pos=pos,
        n_total_records=n_total_records,
        n_kept_records=G.shape[1],
        n_dropped_non_snp=n_dropped_non_snp,
        n_dropped_non_binary=n_dropped_non_binary,
        n_dropped_missing=n_dropped_missing,
        n_dropped_nonseg=n_dropped_nonseg,
    )
