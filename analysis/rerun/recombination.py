#!/usr/bin/env python3
"""Build a fresh fastRho target union and reannotate fixed published BGS fits.

Run prepare after the corrected genome-wide rank/candidate tables exist.
The produced tasks.json contains commands for both GPU shards and CPU checks.
No old recombination predictions or gene-level BGS summaries are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "analysis/recombination"
sys.path.insert(0, str(SCRIPTS))
from infer_maps import EXPECTED, sha, weighted_intervals
from prepare_targets import OVERRIDES
from extract_buffalo_kern import MODEL_SHA, SUMMARY_SHA

FIXED = {**OVERRIDES, "TACR1": "CEU", "NFATC2": "GIH", "ATF7IP": "CHS", "C12orf75": "CDX"}


def target_union(ranks, candidates, old):
    if not ranks.gene_name.is_unique or not ranks.gene_id.is_unique:
        raise ValueError("Gene names and IDs must be unique in refreshed ranks")
    stats = ranks.set_index("gene_name")
    # Keep all explicitly discussed examples and controls. Old Figure-2-only
    # candidates are replaced by the caller's new candidate table(s).
    keep = old.sources.fillna("").str.contains(r"\.tex|neighbour|fixed", regex=True)
    fixed = old[keep | old.gene.isin(FIXED)].set_index("gene")
    names = set(fixed.index) | set(FIXED)
    sources = {gene: {"fixed manuscript example/control"} for gene in names}
    for label, table in candidates:
        if "gene_name" not in table:
            raise ValueError(f"Candidate table needs gene_name: {label}")
        for gene in table.gene_name:
            names.add(gene)
            sources.setdefault(gene, set()).add(label)
    missing = names - set(stats.index)
    if missing:
        raise ValueError(f"Target genes absent from refreshed ranks: {sorted(missing)}")
    rows = []
    for gene in sorted(names):
        row = stats.loc[gene]
        start, end = int(row.start) - 1, int(row.end)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid one-based closed gene coordinates: {gene}")
        lo, hi = max(0, start - 500000), end + 500000
        if gene in ("GRK2", "TREM2") and gene in fixed.index:
            lo, hi = int(fixed.loc[gene].display_start0), int(fixed.loc[gene].display_end0)
        pop = FIXED.get(gene, row.min_pop)
        if not isinstance(pop, str) or not pop:
            raise ValueError(f"No focal population for {gene}")
        rows.append(dict(gene=gene, gene_id=row.gene_id, chrom=f"chr{int(row.chr)}",
            gene_start0=start, gene_end0=end, display_start0=lo, display_end0=hi,
            extract_start0=max(0, lo - 500000), extract_end0=hi + 500000,
            focal_population=pop, populations=",".join(dict.fromkeys([pop, "YRI"])),
            priority=0 if gene in ("GRK2", "TREM2") else 1 if gene in FIXED else 2,
            minimum_tmrca_rank=float(row.min_rank), sources="; ".join(sorted(sources[gene]))))
    return pd.DataFrame(rows).sort_values(["priority", "gene"])


def checked_bgs_reference(reference):
    provenance = json.loads((reference / "buffalo_kern2024_provenance.json").read_text())
    if (provenance["source_model_sha256"] != MODEL_SHA or
            provenance["released_summary_sha256"] != SUMMARY_SHA or
            provenance["assembly"] != "GRCh38"):
        raise ValueError("Unexpected Buffalo-Kern reference fit or assembly")
    for name, digest in provenance["output_sha256"].items():
        if sha(reference / name) != digest:
            raise ValueError(f"Changed fixed BGS reference: {name}")
    if not {"YRI", "CEU"} <= set(provenance["populations"]):
        raise ValueError("Both YRI and CEU fixed BGS maps are required")
    return provenance


def prepare(a):
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(f"Fresh output root required: {output}")
    ranks = pd.read_csv(a.ranks)
    old = pd.read_csv(a.fixed_targets, sep="\t")
    tables = [(p.name, pd.read_csv(p)) for p in a.candidates]
    targets = target_union(ranks, tables, old)
    bundle = a.bundle.resolve()
    if {name: sha(bundle / name) for name in EXPECTED} != EXPECTED:
        raise ValueError("fastRho checkpoint/statistics checksum mismatch")
    bgs = checked_bgs_reference(a.bgs_reference)
    panel = pd.read_csv(a.panel, sep=r"\s+")
    if not panel["sample"].is_unique:
        raise ValueError("Duplicate Phase-3 sample membership")
    if set(",".join(targets.populations).split(",")) - set(panel["pop"]):
        raise ValueError("Requested populations missing from Phase-3 panel")
    # This stage hashes large local VCFs once. Run it through Slurm on Betty.
    input_files = {}
    for chrom in sorted(set(targets.chrom)):
        path = a.vcf_dir.resolve() / f"{chrom}.vcf.gz"
        if not any(Path(str(path) + suffix).is_file() for suffix in (".tbi", ".csi")):
            raise FileNotFoundError(f"Missing VCF index: {path}")
        info = path.stat()
        input_files[str(path)] = dict(sha256=sha(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    reusable = {}
    reuse_audit = []
    for atlas in a.reuse_atlas:
        # Include the corrected-control inputs as separate possible caches.
        for base in (atlas.resolve(), atlas.resolve() / "control_update"):
            approved = {}
            for path in (base / "maps").glob("*.json"):
                record = json.loads(path.read_text())
                if "gene" in record and "input_cache_sha256" in record:
                    approved.setdefault(record["gene"], set()).add(record["input_cache_sha256"])
            for gene in targets.gene:
                source = base / "cache" / f"{gene}.npz"
                if not source.exists():
                    continue
                digest = sha(source)
                if digest not in approved.get(gene, set()):
                    reuse_audit.append(dict(gene=gene, path=str(source), accepted=False,
                                           reason="No matching historical input checksum"))
                    continue
                reusable.setdefault(gene, []).append(dict(path=str(source), sha256=digest))
                reuse_audit.append(dict(gene=gene, path=str(source), accepted=True, sha256=digest))
    (output / "inputs").mkdir(parents=True)
    (output / "bgs").mkdir()
    targets.to_csv(output / "inputs/targets.tsv", sep="\t", index=False)
    shutil.copyfile(a.panel, output / "inputs/phase3.panel")
    shutil.copyfile(a.ranks, output / "inputs/genome_wide_ranks.csv")
    for name in ["buffalo_kern2024_provenance.json", *bgs["output_sha256"]]:
        shutil.copyfile(a.bgs_reference / name, output / "bgs" / name)
    (output / "inputs/reusable_inputs.json").write_text(json.dumps(reusable, indent=2) + "\n")
    (output / "inputs/source_files.json").write_text(json.dumps(input_files, indent=2) + "\n")
    provenance = dict(schema=1, assembly="GRCh38", coordinates="0-based half-open",
        candidate_files={str(p.resolve()): sha(p) for p in a.candidates},
        ranks_sha256=sha(a.ranks), fixed_targets_sha256=sha(a.fixed_targets), panel_sha256=sha(a.panel),
        bundle_sha256=EXPECTED, bgs_source_sha256=sha(a.bgs_reference / "buffalo_kern2024_provenance.json"),
        script_sha256=sha(__file__), n_genes=len(targets),
        n_maps=sum(len(p.split(",")) for p in targets.populations), reuse_audit=reuse_audit,
        rule="Union supplied refreshed candidate tables with fixed manuscript examples/controls; case populations override refreshed min_pop; plus YRI",
        old_predictions_reused=False, fixed_published_BGS_predictions_reused=True)
    (output / "inputs/target_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    tasks = {"gpu": [], "cpu_after_gpu": [], "cpu_independent": []}
    for shard in range(a.shards):
        tasks["gpu"].append([sys.executable, str(SCRIPTS / "infer_maps.py"), "--root", str(output),
            "--bundle", str(bundle), "--shard", str(shard), "--shards", str(a.shards), "--split-check",
            "--reuse-inputs", str(output / "inputs/reusable_inputs.json"), "--vcf-dir", str(a.vcf_dir.resolve()),
            "--source-manifest", str(output / "inputs/source_files.json"), "--require-local-inputs"])
    tasks["cpu_independent"].append([sys.executable, str(Path(__file__).resolve()), "bgs-annotate", "--root", str(output)])
    tasks["cpu_after_gpu"].append([sys.executable, str(SCRIPTS / "summarize_maps.py"), "--root", str(output),
                                  "--ranks", str(output / "inputs/genome_wide_ranks.csv")])
    tasks["cpu_after_gpu"].append([sys.executable, str(SCRIPTS / "validate_run.py"), "--root", str(output)])
    (output / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ("n_genes", "n_maps")}))
    print(output / "tasks.json")


def annotate_bgs(a):
    root = a.root.resolve()
    reference = root / "bgs"
    checked_bgs_reference(reference)
    ranks = pd.read_csv(root / "inputs/genome_wide_ranks.csv")
    targets = pd.read_csv(root / "inputs/targets.tsv", sep="\t")
    output = root / "bgs_annotations"
    output.mkdir(exist_ok=False)
    result = ranks.copy()
    for pop in ("YRI", "CEU"):
        track = pd.read_csv(reference / f"buffalo_kern2024_{pop}_CADD6_100kb.tsv.gz", sep="\t")
        result[f"Bprime_{pop}"] = np.nan
        result[f"Bprime_{pop}_coverage"] = np.nan
        for chrom, genes in result.groupby("chr"):
            bins = track[track.chrom == f"chr{int(chrom)}"]
            starts, ends = genes.start.to_numpy(dtype=np.int64) - 1, genes.end.to_numpy(dtype=np.int64)
            values, covered = weighted_intervals(bins.start.to_numpy(), bins.end.to_numpy(), bins.Bprime.to_numpy(), starts, ends)
            result.loc[genes.index, f"Bprime_{pop}"] = values
            result.loc[genes.index, f"Bprime_{pop}_coverage"] = covered / (ends - starts)
    result.to_csv(output / "all_gene_Bprime.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    result[result.gene_name.isin(targets.gene)].to_csv(output / "target_gene_Bprime.csv", index=False)
    record = dict(fixed_fit_provenance_sha256=sha(reference / "buffalo_kern2024_provenance.json"),
                  ranks_sha256=sha(root / "inputs/genome_wide_ranks.csv"),
                  target_sha256=sha(root / "inputs/targets.tsv"), script_sha256=sha(__file__),
                  genes=len(result), target_genes=len(targets), coordinates="gene input one-based closed; BGS zero-based half-open",
                  interpretation="Length-weighted fixed published reference Bprime with explicit coverage; no gap filling, interpolation, or refitting")
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("ranks", "fixed-targets", "panel", "bundle", "vcf-dir", "bgs-reference", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    prepare_parser.add_argument("--reuse-atlas", type=Path, action="append", default=[])
    prepare_parser.add_argument("--shards", type=int, default=8)
    annotate = sub.add_parser("bgs-annotate")
    annotate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.shards < 1:
            parser.error("shards must be positive")
        prepare(args)
    else:
        annotate_bgs(args)


if __name__ == "__main__":
    main()
