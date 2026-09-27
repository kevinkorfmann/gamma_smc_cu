"""Snapshot real GRCh38 recombination, exon and B tracks for a region."""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import stdpopsim

from common import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chrom", default="22")
    p.add_argument("--left", type=int, default=20_000_000)
    p.add_argument("--right", type=int, default=21_000_000)
    p.add_argument("--map", default="PyrhoYRI_GRCh38")
    p.add_argument("--b-map", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.left < 0 or args.right <= args.left:
        raise ValueError("Invalid interval")
    species = stdpopsim.get_species("HomSap")
    genetic_map = species.get_genetic_map(args.map)
    whole_map = genetic_map.get_chromosome_map(args.chrom)
    rmap = whole_map.slice(args.left, args.right, trim=True)
    if not np.all(np.isfinite(rmap.rate)) or np.any(rmap.rate < 0):
        raise ValueError("Region has missing genetic-map coverage")
    annotation = species.get_annotations("ensembl_havana_104_exons")
    exons = annotation.get_chromosome_annotations(args.chrom)
    exons = exons[(exons[:, 1] > args.left) & (exons[:, 0] < args.right)]
    exons = np.clip(exons - args.left, 0, args.right - args.left)
    if len(exons) == 0:
        raise ValueError("No selected annotation in region")
    # Catalog intervals must be disjoint for the DFE assignment.
    if np.any(exons[1:, 0] < exons[:-1, 1]):
        raise ValueError("Overlapping catalog intervals require an audited union")
    b = pd.read_csv(args.b_map)
    b = b[(b.pos + 500 > args.left) & (b.pos - 500 < args.right)].copy()
    if not len(b) or not np.all(np.isfinite(b.B)) or not b.B.between(0, 1).all():
        raise ValueError("Invalid B map")
    if not np.all(b.pos.to_numpy() % 1000 == 500):
        raise ValueError("Unexpected B-map coordinate convention")
    starts = np.maximum(b.pos.to_numpy() - 500, args.left)
    ends = np.minimum(b.pos.to_numpy() + 500, args.right)
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Duplicate/overlapping B-map bins")
    coverage = (ends - starts).sum() / (args.right - args.left)
    if coverage < .99:
        raise ValueError("Insufficient B coverage")
    np.savez_compressed(args.out / "maps.npz", map_position=rmap.position,
                        map_rate=rmap.rate, exons=exons,
                        b_start=starts-args.left, b_end=ends-args.left, B=b.B.to_numpy())
    meta = dict(assembly="GRCh38", chrom=args.chrom, left=args.left, right=args.right,
                interval_convention="zero-based half-open", length=args.right-args.left,
                map_id=args.map, map_url=genetic_map.url,
                map_archive_sha256=genetic_map.sha256,
                map_total_morgans=rmap.total_mass, mean_r=rmap.mean_rate,
                annotation_id=annotation.id, annotation=str(annotation),
                exon_bp=int(np.diff(exons, axis=1).sum()),
                b_source="Barroso2026/YRI/regulatory/GRCh38/preprint",
                b_upstream_commit="1d44ba129d354272c52aee705987cd2e6abd8a97",
                b_source_sha256=digest(args.b_map), b_coverage=float(coverage),
                mean_B=float(np.average(b.B, weights=ends-starts)),
                stdpopsim=importlib.metadata.version("stdpopsim"),
                maps_sha256=digest(args.out / "maps.npz"))
    write_json(args.out / "region.json", meta)
    print(meta)


if __name__ == "__main__":
    main()
