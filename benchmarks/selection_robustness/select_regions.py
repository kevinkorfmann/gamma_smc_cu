"""Select the prospective panel using maps and coordinates, never TMRCA ranks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import stdpopsim

from common import digest, seed_for, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b-dir", type=Path, required=True)
    p.add_argument("--genes", type=Path, required=True,
                   help="TSV with chr,start0,end,gene_id,gene_name,is_sd; coordinates only")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    genes = pd.read_csv(args.genes, sep="\t", dtype={"chr":str})
    allowed = {"chr", "start0", "end", "gene_id", "gene_name", "is_sd"}
    if set(genes) != allowed:
        raise ValueError("Supply only declared coordinate/SD columns; no selection ranks")
    species = stdpopsim.get_species("HomSap")
    m = species.get_genetic_map("PyrhoYRI_GRCh38")
    annotation = species.get_annotations("ensembl_havana_104_exons")
    rows, input_hashes = [], {str(args.genes): digest(args.genes)}
    exclusions = dict(case_study=0, missing_recombination=0, zero_map_mass=0,
                      missing_B=0, no_complete_nonSD_gene=0, no_exons=0)
    for chrom in map(str, range(1,23)):
        path = args.b_dir/f"chr{chrom}.csv.gz"
        input_hashes[str(path)] = digest(path)
        b = pd.read_csv(path)
        if (not b.pos.is_monotonic_increasing or b.pos.duplicated().any()
                or not np.all(b.pos.to_numpy()%1000==500)
                or not np.all(np.isfinite(b.B)) or not b.B.between(0,1).all()):
            raise ValueError(f"Invalid B track {path}")
        mp = m.get_chromosome_map(chrom)
        exons = annotation.get_chromosome_annotations(chrom)
        g = genes[genes.chr==chrom]
        exclude = g[g.gene_name.isin(["GRK2", "TREML1", "TREML2", "TREM2", "TREM1", "NCR2"])]
        for left in range(0, int(mp.sequence_length)-3_000_000+1, 3_000_000):
            right = left+3_000_000
            core_left, core_right = left+1_000_000, left+2_000_000
            if ((exclude.start0 < right)&(exclude.end > left)).any():
                exclusions["case_study"] += 1
                continue
            overlap=(mp.position[:-1]<right)&(mp.position[1:]>left)
            if not np.all(np.isfinite(mp.rate[overlap])):
                exclusions["missing_recombination"] += 1
                continue
            chunk = mp.slice(left,right,trim=True)
            if chunk.total_mass<=0:
                exclusions["zero_map_mass"] += 1
                continue
            bb = b[(b.pos>=core_left)&(b.pos<core_right)]
            if len(bb)<990:
                exclusions["missing_B"] += 1
                continue
            gg = g[(g.start0>=core_left)&(g.end<=core_right)&(~g.is_sd)]
            if not len(gg):
                exclusions["no_complete_nonSD_gene"] += 1
                continue
            rr = mp.slice(core_left,core_right,trim=True).mean_rate
            exon_bp = np.maximum(0,np.minimum(exons[:,1],core_right)-np.maximum(exons[:,0],core_left)).sum()
            if exon_bp == 0:
                exclusions["no_exons"] += 1
                continue
            rows.append(dict(region_id=f"chr{chrom}_{left}_{right}",chrom=chrom,
                          left=left,right=right,core_left=core_left,core_right=core_right,
                          mean_r=rr,mean_B=bb.B.mean(),B_coverage=len(bb)/1000,
                          exon_fraction=exon_bp/1_000_000,genes=len(gg)))
    df = pd.DataFrame(rows)
    df["r_stratum"] = pd.qcut(df.mean_r,3,labels=False)
    df["B_stratum"] = df.groupby("r_stratum").mean_B.transform(lambda x:pd.qcut(x,3,labels=False)).astype(int)
    # Fixed chromosome split reserved before any method output is available.
    df["split"] = np.where(df.chrom.astype(int).isin([2,5,8,11,14,17,20]),"held_out","development")
    rng=np.random.default_rng(7829001)
    selected=[]
    for r in range(3):
        for b in range(3):
            for split in ["development","held_out"]:
                eligible=df[(df.r_stratum==r)&(df.B_stratum==b)&(df.split==split)]
                if not len(eligible):
                    raise ValueError(f"Empty prespecified stratum {r,b,split}")
                selected.append(eligible.iloc[rng.integers(len(eligible))].to_dict())
    panel=pd.DataFrame(selected)
    df.to_csv(args.out/"eligible_tiles.tsv",sep="\t",index=False)
    panel.to_csv(args.out/"selected_tiles.tsv",sep="\t",index=False)
    write_json(args.out/"provenance.json",dict(region_seed=7829001,assembly="GRCh38",
               inputs_sha256=input_hashes, map_id=m.id,map_url=m.url,map_sha256=m.sha256,
               code_sha256=digest(__file__),eligible_tiles=len(df),selected_tiles=len(panel),
               exclusion_counts=exclusions,
               status="prospective panel; production gated on biological pilot",
               held_out_chromosomes=[2,5,8,11,14,17,20]))
    # A complete prospective workload list; not a claim it has been simulated.
    count=0
    with open(args.out/"proposed_primary_jobs.jsonl","w") as f:
        for row in panel.to_dict("records"):
            for landscape in ["empirical","uniform"]:
                for regime in ["neutral","bgs","positive","bgs_positive","balancing","bgs_balancing"]:
                    for rep in range(20):
                        job=dict(region=row,landscape=landscape,regime=regime,replicate=rep,
                                 diploid_Ne=10000,haplotypes=50,pivot_pairs=50,
                                 mutation_rate=1.29e-8,DFE="Gamma_K17",
                                 burn_in_Ne=10,provisional_scaling_factor=5,
                                 focal_selection_magnitude=.01,
                                 focal_age_generations=(2000 if "positive" in regime else
                                                        20000 if "balancing" in regime else None),
                                 status="planned_not_launched",production_gate="biological QC and cost assessment")
                        job["job_id"]=f"{row['region_id']}_{landscape}_{regime}_{rep:03d}"
                        job["seed_namespace"]=seed_for("primary-v1",job["job_id"])
                        f.write(json.dumps(job)+"\n");count+=1
    print(json.dumps(dict(eligible=len(df),selected=len(panel),planned_simulations=count)))


if __name__=="__main__":
    main()
