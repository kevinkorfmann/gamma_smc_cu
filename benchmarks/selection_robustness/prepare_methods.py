"""Generate matched ASMC quantities and download one commit-pinned cxt checkpoint."""
import argparse
import importlib.metadata
import json
import subprocess
import urllib.request
from pathlib import Path

import msprime
import numpy as np
from asmc.preparedecoding import prepare_decoding

from common import digest, extract_inputs, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--Ne", type=int, default=1000)
    p.add_argument("--mu", type=float, default=1.29e-8)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / "tools" / f"asmc_constant{args.Ne}"
    out.mkdir(exist_ok=False)
    # ASMC's demographic input uses HAPLOID Ne; SLiM's is DIPLOID.
    (out / "prior.demo").write_text(f"0 {2*args.Ne}\n")
    np.savetxt(out / "prior.disc", np.r_[0, np.geomspace(10, 20*args.Ne, 31)])
    ts = msprime.sim_ancestry(25, sequence_length=10_000_000, population_size=args.Ne,
                             recombination_rate=1e-8, random_seed=78291)
    ts = msprime.sim_mutations(ts, rate=args.mu, random_seed=78292)
    ts.dump(out / "independent_neutral_calibration.trees")
    G, _, _ = extract_inputs(ts)
    with open(out / "calibration.frq", "w") as f:
        f.write("CHR SNP A1 A2 MAF NCHROBS\n")
        for j, count in enumerate(G.sum(axis=0)):
            f.write(f"1 snp{j} A G {min(count,50-count)/50:.8f} 50\n")
    dq = prepare_decoding(demography=str(out/"prior.demo"),
                          discretization=str(out/"prior.disc"),
                          frequencies=str(out/"calibration.frq"), samples=50,
                          mutation_rate=args.mu)
    dq.save_decoding_quantities(str(out/"prior"))
    dq.save_intervals(str(out/"prior"))
    write_json(out/"provenance.json", dict(diploid_Ne=args.Ne, haploid_Ne=2*args.Ne,
               mutation_rate=args.mu, haplotypes=50, calibration_seeds=[78291,78292],
               dq_sha256=digest(out/"prior.decodingQuantities.gz"),
               versions={x:importlib.metadata.version(x) for x in ["asmc-asmc", "asmc-preparedecoding"]}))
    cxt = root/"tools/cxt"
    commit = subprocess.check_output(["git", "-C", str(cxt), "rev-parse", "HEAD"], text=True).strip()
    rel = "checkpoints/broad/broad_epoch=1-step=5280.ckpt"
    pointer = subprocess.check_output(["git", "-C", str(cxt), "show", f"{commit}:{rel}"], text=True)
    expected = next(x.split("sha256:",1)[1] for x in pointer.splitlines() if x.startswith("oid "))
    checkpoint = root/"tools/cxt_broad.ckpt"
    url = f"https://media.githubusercontent.com/media/kr-colab/cxt/{commit}/{rel}"
    urllib.request.urlretrieve(url, checkpoint)
    if digest(checkpoint) != expected:
        raise ValueError("cxt checkpoint differs from commit's Git-LFS object")
    write_json(root/"tools/cxt_provenance.json", dict(commit=commit,url=url,checkpoint_sha256=expected))


if __name__ == "__main__":
    main()
