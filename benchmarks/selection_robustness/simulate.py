"""SLiM technical pilot: real human maps, BGS, directional and balancing selection.

The pilot uses constant Ne. Production demographic/trajectory extensions are
not silently approximated here.
"""
import argparse
import contextlib
import importlib.metadata
import json
import time
from pathlib import Path

import msprime
import numpy as np
import stdpopsim

from common import digest, extract_inputs, seed_for, truth_on_grid, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--regime", choices=["neutral", "bgs", "positive", "bgs_positive", "balancing", "bgs_balancing"], required=True)
    p.add_argument("--landscape", choices=["empirical", "uniform"], default="empirical")
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--Ne", type=int, default=1000)
    p.add_argument("--mu", type=float, default=1.29e-8)
    p.add_argument("--scale", type=float, default=5)
    p.add_argument("--burn", type=float, default=5)
    p.add_argument("--age", type=int, default=1000)
    p.add_argument("--s", type=float, default=.01)
    p.add_argument("--slim", required=True)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--technical-smoke", action="store_true")
    args = p.parse_args()
    if not args.technical_smoke:
        raise ValueError("Production is gated on biological QC; this runner currently requires --technical-smoke")
    args.out.mkdir(parents=True, exist_ok=False)
    region = json.loads((args.region / "region.json").read_text())
    maps = np.load(args.region / "maps.npz")
    if digest(args.region / "maps.npz") != region["maps_sha256"]:
        raise ValueError("Region inputs changed")
    L = region["length"]
    if L % 2000 or args.pairs < 1 or args.pairs > 1225:
        raise ValueError("Require a 2-kb-aligned region and valid pair count")
    if args.Ne / args.scale < 25 or args.s * args.scale >= 1 or args.age >= args.burn * args.Ne:
        raise ValueError("Invalid scaled population, fitness or selected history")
    rmap = msprime.RateMap(position=maps["map_position"], rate=maps["map_rate"])
    if args.landscape == "uniform":
        rmap = msprime.RateMap.uniform(L, rmap.mean_rate)
    contig = stdpopsim.Contig.basic_contig(length=L, mutation_rate=args.mu,
                                          recombination_rate=rmap.mean_rate)
    contig.recombination_map = rmap
    if args.regime.startswith("bgs"):
        contig.add_dfe(intervals=maps["exons"], DFE=stdpopsim.get_species("HomSap").get_dfe("Gamma_K17"))
    model = stdpopsim.PiecewiseConstantSize(args.Ne)
    events = []
    focal = L // 2
    if "positive" in args.regime or "balancing" in args.regime:
        contig.add_single_site("focal", coordinate=focal)
        events.append(stdpopsim.DrawMutation(time=args.age, single_site_id="focal", population="pop_0"))
        balancing = "balancing" in args.regime
        events.append(stdpopsim.ChangeMutationFitness(start_time=args.age, end_time=0,
                       single_site_id="focal", population="pop_0",
                       selection_coeff=-args.s if balancing else args.s,
                       dominance_coeff=-1 if balancing else .5))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Simulation random streams differ between factorial cells. Pairs are shared.
    config["seed"] = seed_for("simulation-v1", region["maps_sha256"], args.regime,
                              args.landscape, args.replicate, args.Ne, args.scale,
                              args.burn, args.age, args.s, args.mu)
    config["region"] = region
    config["versions"] = {x: importlib.metadata.version(x) for x in ["stdpopsim", "msprime", "pyslim", "tskit", "numpy"]}
    config["script_sha256"] = digest(__file__)
    write_json(args.out / "config.json", config)
    kwargs = dict(samples={"pop_0": 25}, seed=config["seed"], extended_events=events,
                  slim_scaling_factor=args.scale, slim_burn_in=args.burn,
                  slim_path=args.slim)
    engine = stdpopsim.get_engine("slim")
    # Export script separately; output filename is temporary, biological code identical.
    with open(args.out / "model.slim", "w") as f, contextlib.redirect_stdout(f):
        engine.simulate(model, contig, slim_script=True, **kwargs)
    start = time.perf_counter()
    ts = engine.simulate(model, contig, logfile=str(args.out / "fitness.tsv"), **kwargs)
    seconds = time.perf_counter() - start
    ts.dump(args.out / "truth.trees")
    if ts.num_samples != 50 or ts.time_units != "generations":
        raise ValueError("Sample count or time units mismatch")
    G, pos, dropped = extract_inputs(ts)
    all_pairs = np.array([(a, b) for a in range(50) for b in range(a + 1, 50)])
    rng = np.random.default_rng(seed_for("pairs-v1", args.replicate))
    pairs = all_pairs[rng.permutation(len(all_pairs))[:args.pairs]]
    edges = np.arange(0, L + 1, 2000)
    truth = truth_on_grid(ts, pairs, edges)
    np.savez_compressed(args.out / "input.npz", G=G, positions=pos, pairs=pairs,
                        edges=edges, truth=truth, map_position=rmap.position,
                        map_rate=rmap.rate, mean_r=rmap.mean_rate)
    focal_freq = None
    if events:
        # A drawn site has zero de novo mutation rate. Lost allele => no site.
        focal_freq = 0.0
        for var in ts.variants():
            if var.site.position == focal:
                focal_freq = float(np.mean(var.genotypes != 0))
    t_focal = ts.at(focal)
    max_focal_time = max(t_focal.time(root) for root in t_focal.roots)
    qc = dict(status="technical_smoke_only", simulation_seconds=seconds,
              sites=ts.num_sites, usable_sites=G.shape[1], dropped_sites=dropped,
              samples=ts.num_samples, trees=ts.num_trees, time_units=ts.time_units,
              diversity=ts.diversity(), neutral_expected_diversity=4*args.Ne*args.mu,
              focal_sample_frequency=focal_freq, focal_root_time=max_focal_time,
              focal_root_within_selected_history=bool(max_focal_time < args.age) if events else None,
              fraction_grid_pairs_older_than_forward_phase=float(np.mean(truth > args.Ne*args.burn)),
              selection_outcome=("not_applicable" if not events else
                                 "absent_in_sample" if focal_freq == 0 else
                                 "fixed_in_sample" if focal_freq == 1 else "segregating_in_sample"),
              observed_frequency_is_sample_not_population=True,
              selected_trajectory_recording="not yet implemented; required before production",
              truth_sha256=digest(args.out / "truth.trees"),
              input_sha256=digest(args.out / "input.npz"))
    write_json(args.out / "qc.json", qc)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
