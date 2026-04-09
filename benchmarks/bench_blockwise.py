"""
Benchmark full-sequence FB against blockwise FB variants.

This is intended for the research/blockwise-aggregate-hmm branch, where we want
to compare the current exact baseline against incremental blockwise changes
without changing the scientific output. The script times explicit-pair decodes
through FlowContext directly to avoid Python wrapper noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tmrca_cu import _core

MU = 1.25e-8
RHO = 1e-8
NE = 10000.0
DEFAULT_FLOW_FIELD = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"


def make_random_panel(n_haps: int, n_sites: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 2, size=(n_haps, n_sites), dtype=np.uint8)
    positions = np.arange(1, n_sites + 1, dtype=np.float64)
    return G, positions


def make_pairs(n_haps: int, n_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(n_haps):
        for j in range(i):
            pairs.append((i, j))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def estimate_full_working_set_gb(n_sites: int, n_pairs: int) -> float:
    return 12.0 * n_sites * n_pairs / 1e9


def estimate_block_chunk_gb(block_sites: int, pair_batch_size: int) -> float:
    return 12.0 * block_sites * pair_batch_size / 1e9


def estimate_streamed_total_gb(
    n_sites: int,
    n_pairs: int,
    block_sites: int,
    pair_batch_size: int,
    max_streams: int,
) -> float:
    output_bytes = 4.0 * n_sites * n_pairs
    scratch_bytes = 12.0 * block_sites * pair_batch_size * max_streams
    return (output_bytes + scratch_bytes) / 1e9


def bench(fn, reps: int) -> tuple[list[float], float]:
    times: list[float] = []
    first_entry = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        if out["mean"].size:
            first_entry = float(out["mean"][0, 0])
    return times, first_entry


def run_config(cfg: dict[str, int | str], flow_field_path: str) -> dict[str, object]:
    n_haps = int(cfg["n_haps"])
    n_sites = int(cfg["n_sites"])
    n_pairs = int(cfg["n_pairs"])
    core_block_sites = int(cfg["core_block_sites"])
    flank_sites = int(cfg["flank_sites"])
    pair_batch_size = int(cfg["pair_batch_size"])
    max_streams = int(cfg["max_streams"])
    reps = int(cfg["reps"])

    G, positions = make_random_panel(n_haps, n_sites, seed=int(cfg["seed"]))
    pairs = make_pairs(n_haps, n_pairs)
    ctx = _core.FlowContext(G, positions, NE, MU, RHO, flow_field_path, 0)

    warm_pairs = pairs[: min(32, len(pairs))]
    ctx.run_fb(warm_pairs, mean_only=True)
    ctx.run_fb_blockwise(
        warm_pairs,
        core_block_sites=min(core_block_sites, n_sites),
        flank_sites=min(flank_sites, n_sites // 2),
        pair_batch_size=pair_batch_size,
        max_streams=max_streams,
        mean_only=True,
    )

    full_times, full_checksum = bench(
        lambda: ctx.run_fb(pairs, mean_only=True),
        reps=reps,
    )
    block_single_times, block_single_checksum = bench(
        lambda: ctx.run_fb_blockwise(
            pairs,
            core_block_sites=core_block_sites,
            flank_sites=flank_sites,
            pair_batch_size=pair_batch_size,
            max_streams=1,
            mean_only=True,
        ),
        reps=reps,
    )
    block_streamed_times, block_streamed_checksum = bench(
        lambda: ctx.run_fb_blockwise(
            pairs,
            core_block_sites=core_block_sites,
            flank_sites=flank_sites,
            pair_batch_size=pair_batch_size,
            max_streams=max_streams,
            mean_only=True,
        ),
        reps=reps,
    )

    padded_sites = min(n_sites, core_block_sites + 2 * flank_sites)
    chunk_pairs = n_pairs if pair_batch_size < 0 else min(n_pairs, pair_batch_size)
    return {
        "label": cfg["label"],
        "n_haps": n_haps,
        "n_sites": n_sites,
        "n_pairs": len(pairs),
        "core_block_sites": core_block_sites,
        "flank_sites": flank_sites,
        "pair_batch_size": pair_batch_size,
        "max_streams": max_streams,
        "full_working_set_gb_est": round(estimate_full_working_set_gb(n_sites, len(pairs)), 4),
        "block_single_scratch_gb_est": round(estimate_block_chunk_gb(padded_sites, chunk_pairs), 4),
        "block_streamed_total_gb_est": round(
            estimate_streamed_total_gb(n_sites, len(pairs), padded_sites, chunk_pairs, max_streams),
            4,
        ),
        "full_times_s": [round(t, 4) for t in full_times],
        "block_single_times_s": [round(t, 4) for t in block_single_times],
        "block_streamed_times_s": [round(t, 4) for t in block_streamed_times],
        "full_best_s": round(min(full_times), 4),
        "block_single_best_s": round(min(block_single_times), 4),
        "block_streamed_best_s": round(min(block_streamed_times), 4),
        "streamed_vs_single_speedup": round(min(block_single_times) / min(block_streamed_times), 4),
        "streamed_vs_full_speedup": round(min(full_times) / min(block_streamed_times), 4),
        "first_entry": {
            "full": full_checksum,
            "block_single": block_single_checksum,
            "block_streamed": block_streamed_checksum,
        },
        "first_entry_match": bool(
            abs(full_checksum - block_single_checksum) < 1e-4
            and abs(full_checksum - block_streamed_checksum) < 1e-4
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-field-path", default=DEFAULT_FLOW_FIELD)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--n-haps", type=int, default=None)
    parser.add_argument("--n-sites", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--core-block-sites", type=int, default=8192)
    parser.add_argument("--flank-sites", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=256)
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_haps is not None and args.n_sites is not None and args.n_pairs is not None:
        configs = [
            {
                "label": f"{args.n_sites}_sites_{args.n_pairs}_pairs",
                "n_haps": args.n_haps,
                "n_sites": args.n_sites,
                "n_pairs": args.n_pairs,
                "core_block_sites": args.core_block_sites,
                "flank_sites": args.flank_sites,
                "pair_batch_size": args.pair_batch_size,
                "max_streams": args.max_streams,
                "reps": args.reps,
                "seed": args.seed,
            }
        ]
    else:
        configs = [
            {
                "label": "20k_sites_256_pairs",
                "n_haps": 48,
                "n_sites": 20_000,
                "n_pairs": 256,
                "core_block_sites": 4096,
                "flank_sites": 1024,
                "pair_batch_size": 256,
                "max_streams": 4,
                "reps": args.reps,
                "seed": args.seed,
            },
            {
                "label": "50k_sites_512_pairs",
                "n_haps": 64,
                "n_sites": 50_000,
                "n_pairs": 512,
                "core_block_sites": 8192,
                "flank_sites": 2048,
                "pair_batch_size": 256,
                "max_streams": 4,
                "reps": max(2, args.reps - 1),
                "seed": args.seed + 1,
            },
        ]

    results = [run_config(cfg, args.flow_field_path) for cfg in configs]
    text = json.dumps(results, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
