"""Run one reproducible, physically calibrated stdpopsim comparison.

Outputs are immutable: --output-dir must not already exist. No CUDA import or
external-binary resolution occurs until run(), so CPU-only regressions can
exercise simulation, coordinates, calibration and archived-metric replay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import stdpopsim
from scipy.stats import pearsonr
from bench_inputs import materialize_binary_snp_vcf
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIGS_JSON = HERE / "configs.json"
RETAINED_CONFIG_IDS = tuple(range(9)) + tuple(range(10, 15))
LOG_FLOOR = 1e-10


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def captured(command, cwd=None):
    try:
        p = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=30)
        return {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def source_metadata(repo=REPO):
    """Verify exported snapshots, or record the exact checkout being executed."""
    repo = Path(repo).resolve()
    manifest_path = repo / "SOURCE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        commit = manifest.get("git_commit", "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
            raise ValueError("SOURCE_MANIFEST.json must pin a full 40-character git_commit")
        expected = manifest.get("files")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("SOURCE_MANIFEST.json must contain nonempty files mapping")
        actual = {}
        for name, entry in sorted(expected.items()):
            path = (repo / name).resolve()
            if not path.is_relative_to(repo) or not path.is_file():
                raise ValueError(f"invalid or absent source manifest path: {name}")
            digest = entry.get("sha256") if isinstance(entry, dict) else entry
            actual[name] = sha256(path)
            if actual[name] != digest:
                raise ValueError(f"source manifest checksum mismatch: {name}")
        return {"git_commit": commit, "source_manifest_sha256": sha256(manifest_path),
                "source_manifest_verified": True, "source_sha256": actual}
    if not (repo / ".git").exists():
        raise ValueError("exported source without .git requires SOURCE_MANIFEST.json")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).split(b"\0")
    hashes = {p.decode(): sha256(repo / p.decode()) for p in tracked if p and (repo / p.decode()).is_file() and Path(p.decode()).suffix in {".py", ".cu", ".cpp", ".h", ".hpp", ".toml", ".lock", ".json", ".sh"}}
    return {"git_head": captured(["git", "rev-parse", "HEAD"], repo),
            "git_status": captured(["git", "status", "--porcelain"], repo),
            "source_manifest_verified": False, "source_sha256": hashes}


def environment_metadata(flow_field, gamma_smc_bin):
    versions = {}
    for package in ["numpy", "scipy", "msprime", "stdpopsim", "tskit", "gamma-smc-cu"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = source_metadata()
    binaries = {str(gamma_smc_bin): sha256(gamma_smc_bin), str(flow_field): sha256(flow_field)}
    for path in (REPO / "python/gamma_smc_cu").glob("*.so"):
        binaries[str(path)] = sha256(path)
    return {
        "schema_version": 2, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "versions": versions,
        **source, "binary_sha256": binaries,
        "cpu": captured(["lscpu"]),
        "gpu": captured(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]),
        "nvcc": captured(["nvcc", "--version"]),
        "scheduler": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def true_t_all_pairs(ts, pairs, positions, sample_nodes=None):
    """Truth at exact original zero-based positions, without interpolation."""
    positions = np.asarray(positions, dtype=float)
    if np.any(positions < 0) or np.any(positions >= ts.sequence_length) or np.any(np.diff(positions) <= 0):
        raise ValueError("truth positions must be strictly increasing within the sequence")
    nodes = np.asarray(ts.samples() if sample_nodes is None else sample_nodes)
    result = np.empty((len(positions), len(pairs)), dtype=np.float64)
    start = 0
    for tree in ts.trees():
        stop = int(np.searchsorted(positions, tree.interval.right, side="left"))
        if stop > start:
            result[start:stop] = [tree.tmrca(int(nodes[i]), int(nodes[j])) for i, j in pairs]
        start = stop
        if start == len(positions):
            break
    if start != len(positions):
        raise ValueError("not all evaluation positions received truth")
    return result


def r_log(x, y):
    x, y = np.log(np.maximum(x, LOG_FLOOR)), np.log(np.maximum(y, LOG_FLOOR))
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(pearsonr(x, y).statistic)
    return value if np.isfinite(value) else None


def rmse_log(x, y):
    return float(np.sqrt(np.mean((np.log(np.maximum(x, LOG_FLOOR)) - np.log(np.maximum(y, LOG_FLOOR))) ** 2)))


def qstats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(a)), "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)), "n": len(a)}


def pair_metrics(truth, gpu, cpu, pairs):
    if truth.shape != gpu.shape or truth.shape != cpu.shape:
        raise ValueError("truth and predictions must have identical shapes")
    if not all(np.all(np.isfinite(a)) and np.all(a > 0) for a in [truth, gpu, cpu]):
        raise ValueError("truth and predictions must be finite and positive")
    return [{
        "haplotype_i": int(i), "haplotype_j": int(j),
        "r_gamma_smc_cu": r_log(truth[:, p], gpu[:, p]),
        "r_gsmc": r_log(truth[:, p], cpu[:, p]),
        "r_between_methods": r_log(gpu[:, p], cpu[:, p]),
        "rmse_gamma_smc_cu": rmse_log(truth[:, p], gpu[:, p]),
        "rmse_gsmc": rmse_log(truth[:, p], cpu[:, p]),
        "rmse_between_methods": rmse_log(gpu[:, p], cpu[:, p]),
    } for p, (i, j) in enumerate(pairs)]


def align_reference(positions, means, meta, eval_positions, pairs, physical_mu):
    """Map CPU columns using its saved pair identities and exact site keys."""
    theta = float(meta["scaled_mutation_rate"])
    if not np.isfinite(theta) or theta <= 0 or physical_mu <= 0:
        raise ValueError("positive theta and physical mu are required for generation conversion")
    positions = np.asarray(positions)
    if np.any(np.diff(positions) <= 0):
        raise ValueError("reference positions are not strictly increasing")
    indices = np.searchsorted(positions, eval_positions)
    if np.any(indices >= len(positions)) or not np.array_equal(positions[indices], eval_positions):
        raise ValueError("CPU output does not include every exact evaluation position; no interpolation is permitted")
    layout = [tuple(sorted(map(int, p))) for p in meta["pairs"]]
    if len(layout) != len(set(layout)):
        raise ValueError("duplicate CPU pair identities")
    mapping = {p: i for i, p in enumerate(layout)}
    columns = [mapping[tuple(sorted(p))] for p in pairs]
    return means[np.ix_(indices, columns)].astype(np.float64) * (theta / (2 * physical_mu))


def simulate_config(cfg):
    species = stdpopsim.get_species(cfg["species"])
    model = species.get_demographic_model(cfg["model_id"])
    contig = species.get_contig(length=cfg["seq_len"], mutation_rate=cfg["mu"], recombination_rate=cfg["rho"])
    # engine.simulate already generates mutations: never overlay a second layer.
    return stdpopsim.get_engine("msprime").simulate(model, contig, {cfg["pop"]: cfg["n_hap"] // 2}, seed=cfg["seed"])


def run_reference(vcf_gz, target, binary, flow_field, rho_mu, timeout):
    target.mkdir()
    output = target / "posteriors.zst"
    command = [str(binary), "-i", str(vcf_gz), "-o", str(output), "-t", str(rho_mu), "-f", str(flow_field), "-h"]
    t0 = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    (target / "stdout.txt").write_text(process.stdout)
    (target / "stderr.txt").write_text(process.stderr)
    write_json(target / "command.json", {"command": command, "elapsed_seconds": elapsed, "returncode": process.returncode})
    if process.returncode:
        raise RuntimeError(f"gamma_smc failed; inspect {target}")
    meta = json.loads(Path(str(output) + ".meta").read_text())
    n_pairs, n_sites, chunk = (int(meta[k]) for k in ["num_pairs", "sequence_length", "chunk_size"])
    n_chunks = (n_pairs + chunk - 1) // chunk
    with tempfile.TemporaryDirectory(prefix="gsmc-decode-") as td:
        decoded = Path(td) / "posteriors.bin"
        subprocess.run(["zstd", "-q", "-d", str(output), "-o", str(decoded)], check=True)
        raw = np.memmap(decoded, mode="r", dtype=np.float32, shape=(n_chunks, 2, n_sites, chunk))
        # Same layout as the upstream reader; copy means before the mmap closes.
        alpha = np.concatenate([raw[c, 0] for c in range(n_chunks)], axis=1)[:, :n_pairs]
        beta = np.concatenate([raw[c, 1] for c in range(n_chunks)], axis=1)[:, :n_pairs]
        mean = np.asarray(alpha / np.maximum(beta, LOG_FLOOR)).copy()
        del raw
    return np.asarray(meta["output_positions"]), mean, meta, elapsed


def blockwise_metrics(full, blocked, positions, core_sites):
    dlog = np.abs(np.log(np.maximum(blocked, LOG_FLOOR)) - np.log(np.maximum(full, LOG_FLOOR)))
    indices = np.arange(len(positions))
    boundaries = np.arange(core_sites, len(positions), core_sites)
    distance = np.full(len(positions), len(positions))
    for boundary in boundaries:
        distance = np.minimum(distance, np.abs(indices - boundary))
    groups = {"all_sites": np.ones(len(positions), dtype=bool), "within_64_sites_of_boundary": distance <= 64, "more_than_512_sites_from_boundary": distance > 512}
    stats = {}
    for name, mask in groups.items():
        values = dlog[mask].ravel()
        stats[name] = {"n_values": len(values), "median_abs_log_difference": float(np.median(values)) if len(values) else None,
                       "p99_abs_log_difference": float(np.quantile(values, .99)) if len(values) else None,
                       "max_abs_log_difference": float(values.max()) if len(values) else None}
    stats["n_blocks"] = (len(positions) + core_sites - 1) // core_sites
    return stats


def run(cfg, args, output):
    for tool in ["bgzip", "tabix", "zstd"]:
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH")
    flow = Path(args.flow_field or resolve_flow_field_path(str(HERE))).resolve()
    binary = Path(args.gamma_smc_bin or resolve_gamma_smc_bin(str(HERE))).resolve()
    sys.path.insert(0, str(REPO / "python"))
    import gamma_smc_cu
    write_json(output / "environment.json", environment_metadata(flow, binary))
    write_json(output / "config.json", cfg)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as simulation_warnings:
        ts = simulate_config(cfg)
    t_sim = time.perf_counter() - t0
    ts.dump(output / "simulation.trees")
    vcf = output / "input.vcf"
    prepared = materialize_binary_snp_vcf(ts, str(vcf))
    G, positions = prepared.G, prepared.pos
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    if len(positions) < 2:
        raise ValueError("at least two retained sites are needed")
    counts = {k: v for k, v in asdict(prepared).items() if k.startswith("n_")}
    write_json(output / "preprocessing.json", {**counts, "simulation_warnings": [str(w.message) for w in simulation_warnings], "mutation_layers": 1, "vcf_POS_equals_ts_position_plus": 1})
    np.savez_compressed(output / "observations.npz", G=G, positions=positions, pairs=np.asarray(pairs), site_ids=prepared.site_ids, sample_nodes=prepared.sample_nodes)
    t0 = time.perf_counter()
    with open(str(vcf) + ".gz", "wb") as compressed:
        subprocess.run(["bgzip", "-c", str(vcf)], stdout=compressed, check=True)
    subprocess.run(["tabix", "-p", "vcf", str(vcf) + ".gz"], check=True)
    preparation_seconds = time.perf_counter() - t0

    common = dict(mu=cfg["mu"], rho=cfg["rho"], Ne=10000, pairs=pairs, flow_field_path=str(flow), mean_only=True, auto_estimate_theta=True)
    # Full workload warmup: subsequent API times include preprocessing, cache
    # lookup, allocation, decoding and transfer to host, not file serialization.
    warm = gamma_smc_cu.infer(G, positions, **common)
    metadata = warm.get("metadata", {})
    if metadata.get("time_units") != "generations" or "generations_per_coalescent_unit" not in metadata:
        raise RuntimeError("requires the calibrated inference API with explicit time metadata")
    write_json(output / "gpu_calibration.json", metadata)
    del warm
    gpu_seconds, gpu = [], None
    repeat_max_relative_error = []
    for rep in range(args.gpu_repeats):
        t0 = time.perf_counter()
        result = gamma_smc_cu.infer(G, positions, **common)
        gpu_seconds.append(time.perf_counter() - t0)
        np.testing.assert_array_equal(result["positions"], positions)
        current = np.asarray(result["mean"])
        if gpu is None:
            gpu = current.copy()
        else:
            repeat_max_relative_error.append(float(np.max(np.abs(current - gpu) / np.maximum(gpu, LOG_FLOOR))))
        del result, current
    np.save(output / "gpu_mean_generations.npy", gpu)

    cpu_seconds, cpu = [], None
    cpu_metadata = None
    for rep in range(-1, args.cpu_repeats):
        directory = output / ("cpu_warmup" if rep == -1 else f"cpu_rep{rep:02d}")
        p, means, meta, seconds = run_reference(Path(str(vcf) + ".gz"), directory, binary, flow, cfg["rho"] / cfg["mu"], args.cpu_timeout)
        calibrated = align_reference(p, means, meta, positions, pairs, cfg["mu"])
        # CPU metadata prints theta to ten decimals; allow only this rounding.
        if not np.isclose(float(meta["scaled_mutation_rate"]), metadata["scaled_mutation_rate"], rtol=1e-6, atol=5.1e-11):
            raise ValueError("CPU/GPU scaled mutation rates differ beyond metadata rounding")
        if not np.isclose(float(meta["scaled_recombination_rate"]), metadata["scaled_recombination_rate"], rtol=1e-6, atol=5.1e-11):
            raise ValueError("CPU/GPU scaled recombination rates differ beyond metadata rounding")
        if rep >= 0:
            cpu_seconds.append(seconds)
            if cpu is None:
                cpu, cpu_metadata = calibrated, meta
            elif not np.array_equal(cpu, calibrated):
                raise ValueError("CPU predictions differed between identical repetitions")
        del means, calibrated
    np.save(output / "cpu_mean_generations.npy", cpu)
    truth = true_t_all_pairs(ts, pairs, positions, prepared.sample_nodes)
    np.save(output / "truth_generations.npy", truth)
    metrics = pair_metrics(truth, gpu, cpu, pairs)
    write_json(output / "pair_metrics.json", metrics)

    blockwise = []
    selected = np.unique(np.linspace(0, len(pairs) - 1, min(args.blockwise_pairs, len(pairs)), dtype=int))
    for core in args.blockwise_core_sites:
        for flank in args.blockwise_flanks:
            if len(positions) <= core:
                blockwise.append({"core_sites": core, "flank_sites": flank, "status": "not_multiblock", "reason": "sequence has at most one core block; no boundary claim"})
                continue
            t0 = time.perf_counter()
            blocked = gamma_smc_cu.infer_blockwise(G, positions, **{**common, "pairs": [pairs[i] for i in selected]}, core_block_sites=core, flank_sites=flank)
            seconds = time.perf_counter() - t0
            np.testing.assert_array_equal(blocked["positions"], positions)
            observed = blockwise_metrics(gpu[:, selected], blocked["mean"], positions, core)
            observed.update(core_sites=core, flank_sites=flank, seconds=seconds, pair_indices=selected.tolist(), status="measured")
            overall = observed["all_sites"]
            observed["within_tolerance"] = bool(overall["p99_abs_log_difference"] <= args.blockwise_p99_log_tolerance and overall["max_abs_log_difference"] <= args.blockwise_max_log_tolerance)
            np.savez_compressed(output / f"blockwise_core{core}_flank{flank}.npz", mean_generations=blocked["mean"], blocks=blocked["blocks"], pair_indices=selected)
            blockwise.append(observed)
            print(f"blockwise core={core} flank={flank} p99={overall['p99_abs_log_difference']:.6g} max={overall['max_abs_log_difference']:.6g} within_tolerance={observed['within_tolerance']}", flush=True)
    write_json(output / "blockwise_sensitivity.json", {"p99_abs_log_tolerance": args.blockwise_p99_log_tolerance, "max_abs_log_tolerance": args.blockwise_max_log_tolerance, "comparisons": blockwise})

    t_gpu, t_cpu = float(np.median(gpu_seconds)), float(np.median(cpu_seconds))
    result = {**cfg, "schema_version": 2, "n_sites": len(positions), "n_pairs": len(pairs), "t_sim": t_sim,
        "t_gamma_smc_cu_total": t_gpu, "t_gamma_smc_cu_compute": t_gpu,
        "t_gsmc_total": t_cpu + preparation_seconds, "t_gsmc_compute": t_cpu,
        "speedup_compute": t_cpu / t_gpu, "speedup_total": (t_cpu + preparation_seconds) / t_gpu,
        "gpu_api_seconds": gpu_seconds, "cpu_subprocess_seconds": cpu_seconds,
        "reference_input_compression_index_seconds": preparation_seconds,
        "timing_summary": "median after one full-workload warmup; CPU subprocess includes parsing/initialization/serialization; GPU API includes preprocessing/cache/allocation/decoding/host transfer; neither compute measure is kernel-only",
        "gpu_repeat_max_relative_errors": repeat_max_relative_error, "time_units": "generations",
        "gpu_calibration": metadata,
        "cpu_calibration": {"physical_mu": cfg["mu"], "scaled_mutation_rate": cpu_metadata["scaled_mutation_rate"], "generations_per_coalescent_unit": cpu_metadata["scaled_mutation_rate"] / (2 * cfg["mu"])},
        "truth_alignment": "exact original tree-sequence sites; CPU subset by exact coordinate; no interpolation",
        "status": "ok", "blockwise_tolerance_failed": any(c.get("within_tolerance") is False for c in blockwise),
    }
    for key in ["r_gamma_smc_cu", "r_gsmc", "r_between_methods", "rmse_gamma_smc_cu", "rmse_gsmc", "rmse_between_methods"]:
        stats = qstats([m[key] for m in metrics])
        for suffix in ["median", "q25", "q75"]:
            result[key + "_" + suffix] = stats[suffix]
        result[key + "_n"] = stats["n"]
    result["n_pairs_evaluated_tmrca"] = result["r_gamma_smc_cu_n"]
    result["n_pairs_evaluated_gsmc"] = result["r_gsmc_n"]
    result["n_pairs_evaluated_between"] = result["r_between_methods_n"]
    write_json(output / "result.json", result)
    manifest = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return result


def comma_ints(value):
    if not value:
        return []
    values = [int(v) for v in value.split(",")]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-idx", type=int)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--configs", type=Path, default=CONFIGS_JSON)
    parser.add_argument("--output-dir", type=Path, help="New directory for one config; existing directories are never overwritten")
    parser.add_argument("--results-dir", type=Path, help="Compatibility: create config_NNN under this run root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sequence-length", type=int, help="Explicit smoke-test override, retained in config.json")
    parser.add_argument("--gpu-repeats", type=int, default=3)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--cpu-timeout", type=float, default=1800)
    parser.add_argument("--gamma-smc-bin")
    parser.add_argument("--flow-field")
    parser.add_argument("--blockwise-core-sites", type=comma_ints, default=[16384, 65536])
    parser.add_argument("--blockwise-flanks", type=comma_ints, default=[2048, 8192, 32768])
    parser.add_argument("--blockwise-pairs", type=int, default=20)
    parser.add_argument("--blockwise-p99-log-tolerance", type=float, default=.01)
    parser.add_argument("--blockwise-max-log-tolerance", type=float, default=.1)
    parser.add_argument("--require-blockwise-tolerance", action="store_true", help="Exit 3 after saving complete artifacts if any measured setting exceeds tolerance")
    args = parser.parse_args()
    configs = json.loads(args.configs.read_text())
    if args.list_configs:
        print(",".join(str(c["config_idx"]) for c in configs if c["config_idx"] in RETAINED_CONFIG_IDS))
        return
    if args.config_idx is None or bool(args.output_dir) == bool(args.results_dir):
        parser.error("provide --config-idx and exactly one of --output-dir or --results-dir")
    if min(args.gpu_repeats, args.cpu_repeats, args.blockwise_pairs) < 1:
        parser.error("repeat and pair counts must be positive")
    matches = [c for c in configs if c["config_idx"] == args.config_idx]
    if len(matches) != 1:
        parser.error("config index must uniquely identify a configuration; no positional fallback")
    cfg = dict(matches[0])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.sequence_length is not None:
        if args.sequence_length <= 0:
            parser.error("sequence length must be positive")
        cfg["seq_len"] = args.sequence_length
    output = (args.output_dir or args.results_dir / f"config_{cfg['config_idx']:03d}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "invocation.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "config": cfg})
    try:
        result = run(cfg, args, output)
    except Exception:
        (output / "FAILED.txt").write_text(traceback.format_exc())
        raise
    print(json.dumps({"output": str(output), "status": result["status"], "blockwise_tolerance_failed": result["blockwise_tolerance_failed"]}), flush=True)
    if args.require_blockwise_tolerance and result["blockwise_tolerance_failed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
