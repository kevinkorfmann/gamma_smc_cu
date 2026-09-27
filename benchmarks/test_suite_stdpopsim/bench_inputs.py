import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedInputs:
    G: np.ndarray
    pos: np.ndarray
    site_ids: np.ndarray
    sample_nodes: np.ndarray
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
    original_positions = np.asarray(ts.tables.sites.position)
    if not np.all(np.isfinite(original_positions)) or not np.all(original_positions == np.floor(original_positions)):
        raise ValueError("This benchmark requires integer tree-sequence site positions for exact VCF roundtrip")
    os.makedirs(os.path.dirname(out_vcf_path) or ".", exist_ok=True)
    full_vcf_path = out_vcf_path + ".full"
    with open(full_vcf_path, "w") as f:
        # tskit's default POS equals the site coordinate; VCF is 1-based.
        # This explicit transform makes HTSlib POS-1 equal the original site.
        ts.write_vcf(f, contig_id="chr1", position_transform=lambda x: np.asarray(x) + 1)

    positions = []
    site_haplotypes = []
    site_ids = []
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
            if len(alt) != 1 or ref not in "ACGT" or alt[0] not in "ACGT" or ref == alt[0]:
                n_dropped_non_binary += 1
                continue

            haplotypes = []
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

            if bad_record:
                if missing_record:
                    n_dropped_missing += 1
                else:
                    n_dropped_non_binary += 1
                continue

            if not (0 < sum(haplotypes) < len(haplotypes)):
                n_dropped_nonseg += 1
                continue

            dst.write(line)
            positions.append(float(int(fields[1]) - 1))
            site_ids.append(int(fields[2]))
            site_haplotypes.append(haplotypes)

    os.remove(full_vcf_path)

    if not site_haplotypes:
        raise RuntimeError("no binary SNP records remained after VCF normalization")

    G = np.asarray(site_haplotypes, dtype=np.uint8).T
    pos = np.asarray(positions, dtype=np.float64)
    site_ids = np.asarray(site_ids, dtype=np.int64)
    np.testing.assert_array_equal(pos, original_positions[site_ids])
    # msprime/stdpopsim place samples in diploid individual order. Guard this
    # rather than silently changing the pair-to-node correspondence.
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    expected = ts.genotype_matrix()[site_ids].T
    np.testing.assert_array_equal(G, expected)
    return PreparedInputs(
        G=G,
        pos=pos,
        site_ids=site_ids,
        sample_nodes=sample_nodes,
        n_total_records=n_total_records,
        n_kept_records=G.shape[1],
        n_dropped_non_snp=n_dropped_non_snp,
        n_dropped_non_binary=n_dropped_non_binary,
        n_dropped_missing=n_dropped_missing,
        n_dropped_nonseg=n_dropped_nonseg,
    )
