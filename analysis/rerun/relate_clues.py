#!/usr/bin/env python3
"""Fresh, portable Relate/CLUES2 stages with audited ancestral polarization.

Create a manifest with ``manifest --source-repo OLD_REPO --output NEW_ROOT``.
Run each task listed in its ``tasks`` in stage order: prepare, relate, popsize,
locus. Betty requires Slurm for every run; manifest creation only inventories
paths. The same manifest generator can target Sesame's transferred input tree.
No old genealogy or CLUES result is consumed. Every attempt has a new directory.
Dependencies: Relate 1.2.4, CLUES2, Python numpy/scipy/pandas/numba/biopython/pysam.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid


GROUPS = {"chr2_EUR": (2, ["EUR"], 20000),
          "chr6_EUREAS": (6, ["EUR", "EAS"], 30000),
          "chr11_EURSAS": (11, ["EUR", "SAS"], 30000),
          "chr12_EAS": (12, ["EAS"], 20000),
          "chr20_EURSAS": (20, ["EUR", "SAS"], 30000)}

# Targets are declared before inference. Approximate historical anchors use
# the closest retained, polymorphic, ancestrally resolved SNP within 5 kb;
# TREM2 and LCT targets are exact. Never choose a target by its CLUES score.
LOCI = [
    ("GRK2", "chr11_EURSAS", "GIH", 67407126, 0, "case"),
    ("IFIH1", "chr2_EUR", "IBS", 162482016, 5000, "case"),
    ("BPIFA2", "chr20_EURSAS", "GIH", 31617000, 5000, "case"),
    ("SLC6A15", "chr12_EAS", "CHS", 84887000, 5000, "case"),
    ("CCDC92", "chr12_EAS", "CDX", 124400000, 5000, "case"),
    ("CLEC6A", "chr12_EAS", "CDX", 9730000, 5000, "case"),
    ("LCT", "chr2_EUR", "CEU", 135851076, 0, "positive_control"),
    ("TACR1", "chr2_EUR", "CEU", 75123000, 5000, "neutral_control"),
    ("C11orf65", "chr11_EURSAS", "GIH", 108388000, 5000, "neutral_control"),
    ("NFATC2", "chr20_EURSAS", "GIH", 51475000, 5000, "neutral_control"),
    ("ATF7IP", "chr12_EAS", "CHS", 14434000, 5000, "neutral_control"),
    ("C12orf75", "chr12_EAS", "CDX", 105315000, 5000, "neutral_control"),
] + [(f"TREM2_{p}", "chr6_EUREAS", "IBS", p, 0, "case")
     for p in (41121942, 41137356, 41166068, 41176920, 41189316, 41189932, 41191484)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def text_open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def existing(prefix, extension):
    for suffix in (extension, extension + ".gz"):
        path = Path(str(prefix) + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(str(prefix) + extension)


def create_manifest(a):
    if (not 0 <= a.max_converter_flip_fraction <= 1 or a.mu <= 0 or a.selection_bound <= 0
            or min(a.threads, a.memory_gb, a.popsize_iterations, a.branch_samples, a.df, a.t_cutoff) <= 0):
        raise ValueError("Rates, counts and bounds must be positive; flip tolerance must be in [0,1]")
    source, output = Path(a.source_repo).resolve(), Path(a.output).resolve()
    if output.exists():
        raise FileExistsError(f"New output directory required: {output}")
    old = source / "analysis/relate_clues"
    reference = Path(a.reference_root or old / "input_files/Relate_input_files/GRCh38").resolve()
    vcf_dir = Path(a.vcf_dir or source / "analysis/genome_wide/data").resolve()
    loci = {name: dict(group=group, population=pop, target=pos,
                      max_distance=distance, role=role)
            for name, group, pop, pos, distance, role in LOCI}
    groups = {}
    for name, (chrom, supers, ne) in GROUPS.items():
        groups[name] = dict(chromosome=chrom, superpopulations=supers, initial_N=ne,
            vcf=str(vcf_dir / f"chr{chrom}.vcf.gz"),
            ancestor=str(reference / "human_ancestor_GRCh38" / f"homo_sapiens_ancestor_{chrom}.fa.gz"),
            mask=str(reference / "20160622_genome_mask_GRCh38/StrictMask" / f"20160622.chr{chrom}.mask.fasta.gz"),
            genetic_map=str(reference / "recomb_map" / f"genetic_map_chr{chrom}.txt"))
    manifest = dict(schema=1, output=str(output), source_repo=str(source),
        relate_root=str(Path(a.relate_root or old / "tools/relate_v1.2.4").resolve()),
        clues_root=str(Path(a.clues_root or old / "tools/CLUES2").resolve()),
        samples=str(Path(a.samples or vcf_dir / "samples.txt").resolve()),
        seed=a.seed, mutation_rate=a.mu, years_per_generation=28,
        threads=a.threads, relate_memory_gb=a.memory_gb, popsize_iterations=a.popsize_iterations,
        branch_samples=a.branch_samples, df=a.df, t_cutoff=a.t_cutoff,
        selection_bound=a.selection_bound, integration_points=100,
        max_converter_flip_fraction=a.max_converter_flip_fraction,
        groups=groups, loci=loci,
        tasks={"prepare": list(groups), "relate": list(groups),
               "popsize": sorted({x["group"] + "_" + x["population"] for x in loci.values()}),
               "locus": list(loci)},
        ancestry_policy="FASTA ancestral base must equal exactly one VCF REF/ALT; ambiguous bases fail",
        focal_policy="declared exact target or nearest valid retained SNP within declared distance; no score-based retry",
        converter_policy="persist original and modified vectors; reject flips beyond explicit fraction",
        notes=["IFIH1 is rerun in IBS, matching the manuscript's focal population; old CLUES scripts used CEU.",
               "Each focal population receives its own population-size fit; CDX does not reuse CHS coal rates.",
               "Full chromosome trees are freshly inferred. No old anc/mut/coal/newick/times are reused."])
    required = [Path(manifest["samples"]), Path(manifest["relate_root"]) / "bin/Relate",
                Path(manifest["clues_root"]) / "inference.py"]
    required += [Path(g[k]) for g in groups.values() for k in ("vcf", "ancestor", "mask", "genetic_map")]
    missing = [str(p) for p in required if not p.is_file()]
    missing += [g["vcf"] + " (.tbi or .csi)" for g in groups.values()
                if not any(Path(g["vcf"] + ext).is_file() for ext in (".tbi", ".csi"))]
    if missing:
        raise FileNotFoundError("Missing rerun inputs/tools:\n" + "\n".join(missing))
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    print(output / "manifest.json")
    for stage, tasks in manifest["tasks"].items():
        for task in tasks:
            print(f"{stage}\t{task}")


class Runner:
    def __init__(self, manifest, stage, task):
        self.m = manifest
        self.root = Path(manifest["output"])
        self.stage, self.task = stage, task
        self.state = self.root / "state" / stage / task
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = self.state / "RUNNING"
        self.lock.mkdir()  # Prevent two simultaneous workers for the same task.
        self.work = self.root / "attempts" / stage / task / uuid.uuid4().hex[:12]
        self.work.mkdir(parents=True)
        self.r = Path(manifest["relate_root"])
        self.c = Path(manifest["clues_root"])
        self.env = dict(os.environ, PYTHONHASHSEED=str(manifest["seed"]))
        self.env["PATH"] = str(self.r / "bin") + os.pathsep + self.env.get("PATH", "")
        self.env["PYTHONPATH"] = str(self.c) + os.pathsep + self.env.get("PYTHONPATH", "")

    def command(self, argv, name):
        argv = list(map(str, argv))
        record = dict(argv=argv, cwd=str(self.work), started=time.time())
        log = self.work / (name + ".log")
        with open(log, "w") as f:
            result = subprocess.run(argv, cwd=self.work, env=self.env, stdout=f, stderr=subprocess.STDOUT)
        record.update(finished=time.time(), returncode=result.returncode, log=str(log))
        with open(self.work / "commands.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {log}")

    def dependency(self, stage, task):
        state = self.root / "state" / stage / task / "DONE.json"
        record = json.loads(state.read_text())
        if record["manifest_sha256"] != sha256(self.root / "manifest.json"):
            raise ValueError(f"Dependency was run with a different manifest: {state}")
        return record["outputs"]

    def prepare(self):
        g = self.m["groups"][self.task]
        sample_table = {}
        with open(self.m["samples"]) as f:
            columns = next(f).split()
            for line in f:
                row = dict(zip(columns, line.split()))
                sample_table[row["SampleID"]] = row
        self.command([self.r / "bin/RelateFileFormats", "--mode", "ConvertFromVcf",
                      "--haps", "raw.haps", "--sample", "raw.sample", "-i", g["vcf"][:-7]], "convert")
        sample_ids = [line.split()[0] for line in (self.work / "raw.sample").read_text().splitlines()[2:]]
        if len(sample_ids) != len(set(sample_ids)) or any(s not in sample_table for s in sample_ids):
            raise ValueError("VCF/metadata sample IDs are missing or duplicated")
        with open(self.work / "all.poplabels", "w") as labels, open(self.work / "remove_ids.txt", "w") as remove:
            labels.write("sample population group sex\n")
            for sid in sample_ids:
                row = sample_table[sid]
                labels.write(f"{sid} {row['Population']} {row['Superpopulation']} 0\n")
                if row["Superpopulation"] not in g["superpopulations"]:
                    remove.write(sid + "\n")
        ancestor = self.work / "ancestor.fa"
        with gzip.open(g["ancestor"], "rb") as inp, open(ancestor, "wb") as out:
            shutil.copyfileobj(inp, out)
        prefix = self.work / "prepared"
        self.command([self.r / "scripts/PrepareInputFiles/PrepareInputFiles.sh",
            "--haps", "raw.haps", "--sample", "raw.sample", "--ancestor", ancestor,
            "--mask", g["mask"], "--remove_ids", "remove_ids.txt", "--poplabels", "all.poplabels",
            "-o", prefix], "prepare")
        for ext in (".haps", ".sample", ".dist"):
            src = existing(prefix, ext)
            if src.suffix == ".gz":
                with gzip.open(src, "rb") as inp, open(str(prefix) + ext, "wb") as out:
                    shutil.copyfileobj(inp, out)
        existing(prefix, ".annot")
        existing(prefix, ".poplabels")
        hashes = {k: {"path": g[k], "sha256": sha256(g[k])}
                  for k in ("vcf", "ancestor", "mask", "genetic_map")}
        hashes["samples"] = {"path": self.m["samples"], "sha256": sha256(self.m["samples"])}
        write_json(self.work / "input_hashes.json", hashes)
        return dict(prefix=str(prefix), ancestor=str(ancestor), hashes=str(self.work / "input_hashes.json"))

    def relate(self):
        g = self.m["groups"][self.task]
        prepared = self.dependency("prepare", self.task)["prefix"]
        self.command([self.r / "bin/Relate", "--mode", "All", "-m", self.m["mutation_rate"],
            "-N", g["initial_N"], "--haps", prepared + ".haps", "--sample", prepared + ".sample",
            "--map", g["genetic_map"], "--annot", prepared + ".annot", "--dist", prepared + ".dist",
            "--memory", self.m["relate_memory_gb"], "--seed", self.m["seed"], "-o", "fresh"], "relate")
        prefix = self.work / "fresh"
        existing(prefix, ".anc"), existing(prefix, ".mut")
        return dict(prefix=str(prefix))

    def popsize(self):
        group, pop = self.task.rsplit("_", 1)
        tree = self.dependency("relate", group)["prefix"]
        prepared = self.dependency("prepare", group)["prefix"]
        self.command([self.r / "bin/RelateExtract", "--mode", "SubTreesForSubpopulation",
            "--anc", existing(tree, ".anc"), "--mut", existing(tree, ".mut"),
            "--poplabels", prepared + ".poplabels", "--pop_of_interest", pop, "-o", "subtree"], "extract")
        self.command([self.r / "scripts/EstimatePopulationSize/EstimatePopulationSize.sh",
            "-i", self.work / "subtree", "-o", self.work / "popsize", "-m", self.m["mutation_rate"],
            "--poplabels", self.work / "subtree.poplabels", "--num_iter", self.m["popsize_iterations"],
            "--threads", self.m["threads"], "--years_per_gen", self.m["years_per_generation"],
            "--seed", self.m["seed"], "--noplot"], "popsize")
        prefix = self.work / "popsize"
        existing(prefix, ".anc"), existing(prefix, ".mut"), existing(prefix, ".coal")
        return dict(prefix=str(prefix), poplabels=str(self.work / "subtree.poplabels"))

    def locus(self):
        import numpy as np
        import pysam
        locus = self.m["loci"][self.task]
        group, pop = locus["group"], locus["population"]
        g = self.m["groups"][group]
        prepared = self.dependency("prepare", group)
        population = self.dependency("popsize", group + "_" + pop)
        tree, target, tolerance = population["prefix"], locus["target"], locus["max_distance"]
        # A true focal SNP must be retained in the newly inferred population trees.
        retained = set()
        with text_open(existing(tree, ".mut")) as f:
            next(f)
            for line in f:
                fields = line.split(";")
                bp = int(fields[1])
                if abs(bp - target) <= tolerance:
                    retained.add(bp)
        fasta = prepared["ancestor"]
        if not Path(fasta + ".fai").exists():
            # Each prepare group owns its own ancestor. Creating the index is
            # serialized by an advisory lock for concurrent locus workers.
            import fcntl
            with open(fasta + ".index.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not Path(fasta + ".fai").exists():
                    pysam.faidx(fasta)
        with pysam.FastaFile(fasta) as ancestral, pysam.VariantFile(g["vcf"]) as vcf:
            contig = str(g["chromosome"])
            if contig not in vcf.header.contigs:
                contig = "chr" + contig
            if len(ancestral.references) != 1:
                raise ValueError("Expected one chromosome per ancestral FASTA")
            sample_ids = [row.split()[0] for row in Path(population["poplabels"]).read_text().splitlines()[1:]]
            candidates, rejected = [], []
            for row in vcf.fetch(contig, max(0, target - tolerance - 1), target + tolerance):
                if row.pos not in retained:
                    continue
                try:
                    ancestral_base = ancestral.fetch(ancestral.references[0], row.pos - 1, row.pos).upper()
                    alt = row.alts[0] if row.alts and len(row.alts) == 1 else ""
                    derived_base = polarize(ancestral_base, row.ref, alt)
                    allele_index = (row.ref, alt).index(derived_base)
                    vector = []
                    for sid in sample_ids:
                        gt = row.samples[sid]["GT"]
                        if len(gt) != 2 or any(a not in (0, 1) for a in gt) or not row.samples[sid].phased:
                            raise ValueError(f"Incomplete or unphased diploid GT: {sid}")
                        vector.extend(int(a == allele_index) for a in gt)
                    if not 0 < sum(vector) < len(vector):
                        raise ValueError("Focal population is monomorphic")
                    candidates.append(dict(position=row.pos, ref=row.ref, alt=alt, ancestral=ancestral_base,
                        derived=derived_base, derived_count=sum(vector), haplotypes=len(vector),
                        derived_frequency=sum(vector) / len(vector), vector=vector))
                except ValueError as e:
                    rejected.append(dict(position=row.pos, reason=str(e)))
        write_json(self.work / "candidate_audit.json", dict(target=target, rejected=rejected,
            candidates=[{k:v for k,v in c.items() if k != "vector"} for c in candidates]))
        if not candidates:
            raise ValueError("No retained, polymorphic, ancestrally resolved target within declared distance")
        focal = min(candidates, key=lambda c: (abs(c["position"] - target), c["position"]))
        pos = focal["position"]
        # Sample a flanking region for mutation information, then retain only
        # the focal tree for each MCMC draw. The old scripts mixed local trees.
        self.command([self.r / "scripts/SampleBranchLengths/SampleBranchLengths.sh",
            "-i", tree, "-o", self.work / "sampled", "-m", self.m["mutation_rate"],
            "--coal", tree + ".coal", "--num_samples", self.m["branch_samples"],
            "--first_bp", max(1, pos - 150000), "--last_bp", pos + 150000,
            "--format", "n", "--seed", self.m["seed"]], "sample")
        with open(self.work / "sampled.sites") as f:
            names = next(f).split()[1:]
            next(f)
            observed = [line.split()[1] for line in f if int(line.split()[0]) == pos]
        if len(observed) != 1 or names != list(map(str, range(len(focal["vector"])))):
            raise ValueError("Sampled sites have missing/duplicate target or unexpected haplotype order")
        if set(observed[0]) - {focal["ref"], focal["alt"]}:
            raise ValueError("Sampled alleles disagree with VCF REF/ALT")
        sampled_vector = [int(base == focal["derived"]) for base in observed[0]]
        if sampled_vector != focal["vector"]:
            raise ValueError("Sampled haplotype polarization/order disagrees with VCF")
        n_trees = select_focal_trees(self.work / "sampled.newick", self.work / "focal.newick", pos)
        if n_trees != self.m["branch_samples"]:
            raise ValueError(f"Expected {self.m['branch_samples']} focal trees, found {n_trees}")
        np.savetxt(self.work / "derived_original.txt", sampled_vector, fmt="%d")
        focal.update(target=target, population=pop, chromosome=g["chromosome"],
            sample_ids=sample_ids, ancestor_sha256=sha256(g["ancestor"]), n_focal_trees=n_trees)
        write_json(self.work / "polarization.json", focal)
        self.command([sys.executable, self.c / "RelateToCLUES.py", "--RelateSamples", "focal.newick",
            "--DerivedFile", "derived_original.txt", "--out", "converted"], "convert_clues")
        modified = self.work / "converted_derived.txt"
        converter_vector = np.loadtxt(modified, dtype=int).reshape(-1).tolist() if modified.exists() else sampled_vector
        if len(converter_vector) != len(sampled_vector) or set(converter_vector) - {0, 1}:
            raise ValueError("Invalid converter-derived vector")
        flips = [i for i, (a, b) in enumerate(zip(sampled_vector, converter_vector)) if a != b]
        focal.update(converter_flipped_leaves=flips, converter_flip_fraction=len(flips) / len(sampled_vector),
                     converter_derived_frequency=sum(converter_vector) / len(converter_vector))
        write_json(self.work / "polarization.json", focal)
        if focal["converter_flip_fraction"] > self.m["max_converter_flip_fraction"]:
            raise ValueError("CLUES converter changed the observed derived vector beyond the declared tolerance")
        # CLUES2 uses numpy.random for trajectory integration but has no seed
        # option. Seed both RNGs in a clean interpreter before runpy executes it.
        launcher = self.work / "seeded_clues.py"
        launcher.write_text("import random, runpy, sys\nimport numpy as np\n"
            f"random.seed({self.m['seed']}); np.random.seed({self.m['seed']})\n"
            f"sys.argv[0] = {str(self.c / 'inference.py')!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")
        self.command([sys.executable, launcher, "--times", "converted_times.txt", "--coal", tree + ".coal",
            "--popFreq", focal["derived_frequency"], "--tCutoff", self.m["t_cutoff"],
            "--df", self.m["df"], "--CI", 0.95, "--sMax", self.m["selection_bound"],
            "--integration_points", self.m["integration_points"], "--out", "result"], "inference")
        with open(self.work / "result_inference.txt") as f:
            header, values = next(f).split(), next(f).split()
        estimates = {key: float(value) for key, value in zip(header, values)}
        write_json(self.work / "selection_diagnostics.json", dict(
            estimates=estimates, selection_bound=self.m["selection_bound"],
            near_bound=any(abs(value) >= 0.95 * self.m["selection_bound"]
                           for key, value in estimates.items() if key.startswith("SelectionMLE"))))
        summarize_trajectory(self.work / "result", self.m["years_per_generation"])
        return dict(prefix=str(self.work / "result"), polarization=str(self.work / "polarization.json"),
                    trajectory=str(self.work / "result_trajectory95.tsv"))


def polarize(ancestor, ref, alt):
    if any(len(x) != 1 or x not in "ACGT" for x in (ancestor, ref, alt)) or ref == alt:
        raise ValueError("Ambiguous ancestral base or non-biallelic SNP")
    if ancestor == ref:
        return alt
    if ancestor == alt:
        return ref
    raise ValueError("Ancestral base matches neither VCF REF nor ALT")


def select_focal_trees(source, output, position):
    """Relate newick intervals use [chromStart, chromEnd), in .mut coordinates."""
    selected, samples = [], set()
    with open(source) as f:
        header = next(f)
        for line in f:
            fields = line.split("\t")
            if int(fields[1]) <= position < int(fields[2]):
                if fields[3] in samples:
                    raise ValueError("Multiple focal trees for the same MCMC sample")
                samples.add(fields[3])
                selected.append(line)
    with open(output, "w") as f:
        f.write(header)
        f.writelines(selected)
    return len(selected)


def summarize_trajectory(prefix, generation_time):
    import numpy as np
    freqs = np.loadtxt(str(prefix) + "_freqs.txt", delimiter=",")
    posterior = np.loadtxt(str(prefix) + "_post.txt", delimiter=",", ndmin=2)
    if posterior.shape[0] != len(freqs) or np.any(~np.isfinite(posterior)) or np.any(posterior < 0):
        raise ValueError("Malformed CLUES posterior grid")
    totals = posterior.sum(axis=0)
    if np.any(totals <= 0) or not np.allclose(totals, 1, atol=1e-5):
        raise ValueError("CLUES posterior grid is not normalized")
    posterior = posterior / totals
    with open(str(prefix) + "_trajectory95.tsv", "w") as f:
        f.write("generations_before_present\tyears_before_present\tmean\tmedian\tlower_95\tupper_95\n")
        for t in range(posterior.shape[1]):
            cdf = np.cumsum(posterior[:, t])
            quantiles = [freqs[min(np.searchsorted(cdf, q), len(freqs) - 1)] for q in (0.5, 0.025, 0.975)]
            values = [t, t * generation_time, float(freqs @ posterior[:, t]), *quantiles]
            f.write("\t".join(map(str, values)) + "\n")


def run_task(a):
    m = json.loads(Path(a.manifest).read_text())
    if a.task not in m["tasks"][a.stage]:
        raise ValueError("Unknown task for requested stage")
    if "betty" in platform.node().lower() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run Betty computation through Slurm")
    done = Path(m["output"]) / "state" / a.stage / a.task / "DONE.json"
    if done.exists():
        if json.loads(done.read_text())["manifest_sha256"] != sha256(a.manifest):
            raise ValueError("Completed task uses a different manifest; create a new output root")
        print(f"Already completed: {done}")
        return
    runner = Runner(m, a.stage, a.task)
    record = dict(stage=a.stage, task=a.task, hostname=platform.node(), started=time.time(),
                  manifest_sha256=sha256(a.manifest), script_sha256=sha256(__file__),
                  python=sys.version, slurm_job_id=os.environ.get("SLURM_JOB_ID"), work=str(runner.work))
    tool_files = [runner.r / "bin/Relate", runner.r / "bin/RelateExtract", runner.r / "bin/RelateCoalescentRate",
                  runner.r / "bin/RelateFileFormats", runner.r / "bin/RelateMutationRate",
                  runner.r / "scripts/PrepareInputFiles/PrepareInputFiles.sh",
                  runner.r / "scripts/EstimatePopulationSize/EstimatePopulationSize.sh",
                  runner.r / "scripts/SampleBranchLengths/SampleBranchLengths.sh",
                  runner.c / "inference.py", runner.c / "RelateToCLUES.py", runner.c / "hmm_utils.py"]
    try:
        record["tool_sha256"] = {str(p): sha256(p) for p in tool_files}
        record["packages"] = {}
        for package in ("numpy", "scipy", "pandas", "numba", "biopython", "pysam"):
            try:
                record["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                record["packages"][package] = None
        record["outputs"] = getattr(runner, a.stage)()
        record.update(status="complete", finished=time.time())
        write_json(runner.work / "status.json", record)
        write_json(done, record)
        print(done)
    except BaseException as e:
        record.update(status="failed", error=repr(e), finished=time.time())
        write_json(runner.work / "status.json", record)
        raise
    finally:
        runner.lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--source-repo", required=True)
    manifest.add_argument("--output", required=True)
    for name in ("relate-root", "clues-root", "reference-root", "vcf-dir", "samples"):
        manifest.add_argument("--" + name)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--mu", type=float, default=1.25e-8)
    manifest.add_argument("--threads", type=int, default=16)
    manifest.add_argument("--memory-gb", type=int, default=128)
    manifest.add_argument("--popsize-iterations", type=int, default=10)
    manifest.add_argument("--branch-samples", type=int, default=200)
    manifest.add_argument("--df", type=int, default=200)
    manifest.add_argument("--t-cutoff", type=int, default=2000)
    manifest.add_argument("--selection-bound", type=float, default=0.1)
    manifest.add_argument("--max-converter-flip-fraction", type=float, default=0.0)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--stage", choices=("prepare", "relate", "popsize", "locus"), required=True)
    run.add_argument("--task", required=True)
    args = parser.parse_args()
    create_manifest(args) if args.action == "manifest" else run_task(args)


if __name__ == "__main__":
    main()
