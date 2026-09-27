#!/usr/bin/env python3
"""Fetch pinned human B maps and annotate GRCh38 intervals.

The source predicts B at 1-kb window centers. We approximate it as constant
within each corresponding window, without interpolation across missing bins.
All internal intervals and exported BEDs are zero-based, half-open.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/bgs/barroso2026"
COMMIT = "1d44ba129d354272c52aee705987cd2e6abd8a97"
UPSTREAM = "https://raw.githubusercontent.com/nwcol/bgs_lmr/" + COMMIT
MODELS = ("regulatory", "phastcons")
THRESHOLDS = (0.8, 0.9, 0.95)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(cache):
    """Download only public reference data; verify pinned Git blob hashes."""
    cache.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/nwcol/bgs_lmr/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(urllib.request.urlopen(tree_url, timeout=60).read())
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream Git tree")
    hashes = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    paths = []
    for model in MODELS:
        for chrom in range(1, 23):
            remote = ("models/B_maps_1kb/equilibrium_granular_Ne/"
                      f"split_cds_{model}/roulette/B_map_YRI_chr{chrom}_1kb.csv.gz")
            paths.append((remote, f"{model}/chr{chrom}.csv.gz"))
    for p in ("README.md", "data/README.md", "data/recombination_maps/README.txt",
              "tools/py_scripts/build_B_map.py", "tools/py_scripts/README.md",
              "config/config.yaml"):
        paths.append((p, "source_docs/" + p))

    def fetch(item):
        remote, local = item
        dest = cache / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = dest.read_bytes() if dest.exists() else urllib.request.urlopen(
            UPSTREAM + "/" + remote, timeout=120).read()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != hashes[remote]:
            raise ValueError(f"Upstream checksum mismatch: {remote}")
        if not dest.exists():
            temp = dest.with_suffix(dest.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(dest)
        return {"path": local, "upstream_path": remote, "url": UPSTREAM + "/" + remote,
                "bytes": len(data), "git_blob_sha1": blob,
                "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, paths))
    manifest = {"citation": "Barroso et al. (2026), bioRxiv, preprint",
                "doi": "10.64898/2026.06.02.727906", "upstream_commit": COMMIT,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "assembly": "GRCh38", "population": "YRI",
                "models": list(MODELS), "mutation_model": "roulette",
                "demography": "equilibrium_granular_Ne", "files": files}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} files ({sum(f['bytes'] for f in files):,} bytes): {cache}", flush=True)


def read_track(cache, model, chrom):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(cache / model / f"chr{chrom}.csv.gz")
    pos = frame.pos.to_numpy(dtype=np.int64)
    b = frame.B.to_numpy(dtype=float)
    if not (frame.chrom.eq(chrom).all() and np.all(pos >= 500)
            and np.all(pos % 1000 == 500) and np.all(np.diff(pos) >= 1000)):
        raise ValueError(f"Unexpected coordinate grid: {model}, chr{chrom}")
    if not (np.isfinite(b).all() and np.all((b >= 0) & (b <= 1))):
        raise ValueError(f"Invalid B values: {model}, chr{chrom}")
    return pos - 500, pos + 500, b


def summarize_interval(starts, ends, values, start, end, thresholds=THRESHOLDS):
    """Length-weighted B summary with explicit coverage, including map gaps."""
    import numpy as np
    if end <= start or start < 0:
        raise ValueError(f"Invalid half-open interval [{start}, {end})")
    lo = np.searchsorted(ends, start, side="right")
    hi = np.searchsorted(starts, end, side="left")
    w = np.maximum(0, np.minimum(ends[lo:hi], end) - np.maximum(starts[lo:hi], start))
    v = values[lo:hi]
    covered = int(w.sum())
    out = {"coverage": covered / (end - start), "covered_bp": covered,
           "mean": float(np.average(v, weights=w)) if covered else np.nan,
           "min": float(v.min()) if covered else np.nan}
    for threshold in thresholds:
        # Denominator is mapped bp, not full gene length. Coverage is separate.
        out[f"fraction_lt_{threshold:g}"] = float(w[v < threshold].sum() / covered) if covered else np.nan
    return out


def merged_intervals(starts, ends, keep):
    """Merge selected adjacent bins, never crossing a gap or excluded bin."""
    current = None
    for start, end, selected in zip(starts, ends, keep):
        if selected:
            if current is not None and current[1] == start:
                current[1] = int(end)
            else:
                if current is not None:
                    yield tuple(current)
                current = [int(start), int(end)]
        elif current is not None:
            yield tuple(current)
            current = None
    if current is not None:
        yield tuple(current)


def annotate(args):
    import numpy as np
    import pandas as pd
    cache, out = args.cache, args.output
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["upstream_commit"] != COMMIT or manifest["assembly"] != "GRCh38":
        raise ValueError("Unexpected source manifest")
    for item in manifest["files"]:
        if sha256(cache / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed reference: {item['path']}")
    genes = pd.read_csv(args.genes)
    if not genes.gene_id.is_unique or not genes.chr.between(1, 22).all():
        raise ValueError("Expected unique autosomal gene IDs")
    coords = genes[["start", "end"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all():
        raise ValueError("Gene coordinates must be finite integers")
    sd = pd.read_csv(args.sd).set_index("gene_id").is_sd
    if not sd.index.is_unique or not genes.gene_id.isin(sd.index).all() or not sd.isin([True, False]).all():
        raise ValueError("Missing, duplicate, or invalid SD flags")
    genes["is_sd"] = genes.gene_id.map(sd).astype(bool)
    out.mkdir(parents=True, exist_ok=True)
    pop_columns = [c for c in genes if c.endswith("_rank") and c not in ("min_rank", "max_rank")]
    sensitivity, diagnostics, map_qc = [], [], []
    for model in MODELS:
        summaries = {}
        mask_dir = out / "masks" / model
        mask_dir.mkdir(parents=True, exist_ok=True)
        handles = {}
        for threshold in THRESHOLDS:
            for kind in ("exclude_B_lt", "keep_B_ge"):
                handles[kind, threshold] = gzip.open(mask_dir / f"{kind}_{threshold:g}.bed.gz", "wt")
        try:
            for chrom in range(1, 23):
                starts, ends, b = read_track(cache, model, chrom)
                map_qc.append({"model": model, "chrom": chrom, "bins": len(b),
                               "mapped_bp": int((ends-starts).sum()), "min_B": float(b.min()),
                               "max_B": float(b.max()), "gap_count": int((starts[1:] != ends[:-1]).sum())})
                for threshold in THRESHOLDS:
                    for kind, keep in (("exclude_B_lt", b < threshold), ("keep_B_ge", b >= threshold)):
                        for start, end in merged_intervals(starts, ends, keep):
                            handles[kind, threshold].write(f"chr{chrom}\t{start}\t{end}\n")
                for gene in genes[genes.chr == chrom].itertuples():
                    start = int(gene.start) - (args.gene_coordinates == "one-based-closed")
                    summaries[gene.gene_id] = summarize_interval(starts, ends, b, start, int(gene.end))
                print(f"Annotated {model} chr{chrom}", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        annotations = pd.DataFrame.from_dict(summaries, orient="index").add_prefix(f"b_{model}_")
        genes = genes.join(annotations, on="gene_id", validate="one_to_one")
        eligible = (~genes.is_sd & genes[f"b_{model}_coverage"].ge(args.min_coverage)
                    & genes[pop_columns].notna().any(axis=1))
        decile_col = f"b_{model}_decile"
        genes[decile_col] = np.nan
        genes.loc[eligible, decile_col] = pd.qcut(genes.loc[eligible, f"b_{model}_mean"], 10,
                                                labels=False, duplicates="drop") + 1
        for col in pop_columns:
            pop = col.removesuffix("_rank")
            valid = eligible & genes[col].notna()
            # Descriptive within-B-stratum percentile, not calibrated significance.
            conditional = f"{pop}_{model}_within_B_decile_rank"
            genes[conditional] = np.nan
            genes.loc[valid, conditional] = genes.loc[valid].groupby(decile_col)[col].rank(pct=True)
            diagnostics.append({"model": model, "population": pop, "n": int(valid.sum()),
                                "spearman_B_vs_original_rank": float(genes.loc[valid, f"b_{model}_mean"].corr(genes.loc[valid, col], method="spearman"))})
            original_top = valid & genes[col].le(.01)
            for threshold in THRESHOLDS:
                keep = valid & genes[f"b_{model}_mean"].ge(threshold)
                rerank_col = f"{pop}_{model}_meanB_ge_{threshold:g}_rank"
                genes[rerank_col] = np.nan
                genes.loc[keep, rerank_col] = genes.loc[keep, col].rank(pct=True)
                sensitivity.append({"model": model, "population": pop, "threshold": threshold,
                                    "eligible_genes": int(valid.sum()), "kept_genes": int(keep.sum()),
                                    "original_top1_genes": int(original_top.sum()),
                                    "original_top1_kept": int((keep & original_top).sum()),
                                    "filtered_top1_genes": int(genes[rerank_col].le(.01).sum())})
    genes.to_csv(out / "gene_bgs_annotations.csv.gz", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "mask_sensitivity.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "population_diagnostics.csv", index=False)
    pd.DataFrame(map_qc).to_csv(out / "map_validation.csv", index=False)
    genes[genes.gene_name.isin(["GRK2", "TREML1", "TREM2", "SLC24A5", "LCT", "EDAR", "FADS1"])].to_csv(out / "candidate_bgs_annotations.csv", index=False)
    provenance = {"source_manifest_sha256": sha256(cache / "manifest.json"),
                  "source_commit": COMMIT, "assembly": "GRCh38", "gene_coordinates": args.gene_coordinates,
                  "genes_sha256": sha256(args.genes), "sd_sha256": sha256(args.sd),
                  "script_sha256": sha256(__file__), "min_coverage": args.min_coverage,
                  "thresholds": list(THRESHOLDS), "gene_count": len(genes),
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "interpretation": "Exploratory annotations and gene filtering; no TMRCA re-estimation or calibrated significance."}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--cache", type=Path, default=CACHE)
    ann = sub.add_parser("annotate")
    ann.add_argument("--cache", type=Path, default=CACHE)
    ann.add_argument("--genes", type=Path, required=True)
    ann.add_argument("--sd", type=Path, required=True)
    ann.add_argument("--gene-coordinates", choices=["one-based-closed", "zero-based-half-open"], required=True)
    ann.add_argument("--min-coverage", type=float, default=.95)
    ann.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.command == "download":
        download(args.cache)
    else:
        if not 0 < args.min_coverage <= 1:
            parser.error("--min-coverage must be in (0,1]")
        annotate(args)


if __name__ == "__main__":
    main()
