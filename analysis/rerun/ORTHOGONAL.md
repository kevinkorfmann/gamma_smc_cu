# Portable orthogonal-analysis rerun

`orthogonal.py` runs one explicit task at a time. It never submits jobs. Betty
GPU and analysis tasks belong in Slurm allocations. Choose a new output root.
The default input layout is `INPUTS/cache/{parsed,genes}` and
`INPUTS/samples.txt`; override `--cache-dir`, `--genes-dir`, or `--samples` if
staged elsewhere. Inputs and outputs must be separate non-nested roots.

Generate the task manifest without requiring installed scientific binaries:

```sh
python analysis/rerun/orthogonal.py manifest \
  --inputs-root /data/run-inputs --output-root /runs/run-id/orthogonal \
  --tools-root /opt/run-tools
```

Read `OUTPUT/manifest/tasks.tsv`. Each row's `action`, `chr`, `pop`, and `gene`
become CLI arguments (omit empty fields). For example:

```sh
python analysis/rerun/orthogonal.py selscan --chr 22 --pop YRI \
  --inputs-root /data/run-inputs --output-root /runs/run-id/orthogonal \
  --tools-root /opt/run-tools --threads 8

python analysis/rerun/orthogonal.py regional --chr 6 --pop IBS --gene TREM2 \
  --inputs-root /data/run-inputs --output-root /runs/run-id/orthogonal \
  --tools-root /opt/run-tools --core-block-sites 65536 --flank-sites 8192
```

The default manifest contains 1,433 tasks: 572 selscan, 572 gene-body H12,
66 neighborhood-background, 60 neighborhood, 36 regional Gamma-SMC/cxt,
36 ASMC, 26 normalization, 26 gene-body H12 aggregation, 18 variant,
18 XP-EHH and 3 neighborhood aggregation tasks. The 18 fixed target
combinations comprise seven case studies (including IFIH1 and TREM2), five
positive controls, five SI controls, and C11orf65. Regional and ASMC tasks
include the focal population and YRI. TREM2 neighborhood H12 is evaluated in
all 26 populations. Use `--targets` with a TSV (`gene, chr, pop, group` columns)
to override the target set. The saved controls retain their historical
identities; rerun TMRCA ranks must establish whether they still meet the
intended control criterion.

Dependencies:

- `normalize --pop POP`: all 22 matching `selscan` tasks.
- `h12-aggregate --pop POP`: all 22 matching `h12` tasks.
- `neighborhood-aggregate --pop POP`: all 22 matching neighborhood-background
  tasks; default backgrounds are GIH, CDX and CHS, matching the SI percentiles.
- `asmc --gene GENE --pop POP --chr CHR`: matching `regional` task.

All other tasks can run independently. Every task records `COMPLETE.json`
with configuration, input/source identity and output checksums. A method
failure creates `FAILED.json` and a nonzero exit code; no partial ASMC/cxt
result is accepted as successful. `--resume` skips only matching completed
outputs; failed/incomplete directories must first be inspected and moved
aside explicitly. Dependency outputs are checksum-verified when consumed.

## Scientific conventions

- Shared cache loader requires the strictly increasing, complete phased
  binary SNP stream prepared by the input pipeline. All haplotype membership
  follows cached sample IDs, never sample-file row order. True NPY memory
  mapping is supported. Population/region slicing happens before copying.
- selscan iHS/nSL use focal-population polymorphic sites, explicit MAF 0.05
  and the uniform 1 cM/Mb map. ALT/REF coding is retained; no ancestral-state
  assertion is made. The binary path defaults to
  `TOOLS/selscan/bin/linux/selscan`; override `--selscan-bin` if necessary.
- Normalization pools all 22 chromosomes *within each population*, in 20
  ALT-frequency bins (width 0.05, at least 50 finite scores, population
  standard deviation). This explicitly changes the historical per-chromosome
  normalization. Per-gene counts include only finite normalized scores.
  Missing gene-level scores remain missing, including their ranks. Full
  normalized per-site tracks and bin parameters are retained.
- Gene-body H12 drops focal-population monomorphic sites, uses 400-site
  windows stepped by 50, and takes the maximum whose central SNP lies in the
  gene. All genes remain represented; absent windows produce NaN. Percentiles
  use the weak empirical CDF among finite summaries.
- Neighborhood H12 retains all cohort SNP sites, including monomorphic sites
  in the focal population, and restarts windows at each midpoint ±500 kb.
  Background tasks compute the same statistic for every annotated gene so
  target/control percentiles have the proper denominator. H2/H1 is recorded
  at the peak-H12 window and equals zero for a single unique haplotype.
- Variant summaries use midpoint ±500 kb, comparing the focal population
  against pooled non-focal-superpopulation haplotypes. Retained site tables
  also contain ALT frequencies for all 26 populations. D/E counts exclude
  ties, and Hudson FST retains finite-sample numerator corrections.
- XP-EHH reproduces the historical pooled-superpopulation versus YRI
  contrast, gene body plus ±500 kb reporting window and additional 2 Mb
  integration context, using raw scikit-allel scores. Its historical allele
  filter is retained and documented in code. A pooled AFR comparison that
  includes YRI is rejected because its samples overlap the reference.
- Regional Gamma-SMC uses full-chromosome theta calibration, physical
  mutation/recombination rates, fixed block/flank settings and the same
  physical-generation convention as the genome-wide rerun. Pair seed42
  selects20 pairs. cxt broad uses50 haplotypes, subset seed123,3 replicates,
  CPU inference; cxt/torch must be installed in the run environment.
- ASMC uses the same pivot pairs and a deterministic100-haplotype subset
  as the retained regional approximation. This observed subset size is independent
  of the corrected CEU CSFS grid’s 50 haploid samples. Its default resource is
  `TOOLS/asmc_data/CEU_csfs50/decoding.decodingQuantities.gz`, overridable through
  `--decoding-quantities`. `asmc.asmc` must be importable. Use the freshly generated decoding quantities
  with the CEU history rescaled consistently to mutation rate 1.25e-8. Focal and YRI
  estimates must all succeed; unavailable methods are failures, not NaN
  outputs silently inserted into a supposedly complete comparison.

Scientific Python dependencies: numpy, pandas, scikit-allel; regional tasks
add torch/cxt and the CUDA gamma_smc_cu build; ASMC adds its Python package
and decoding quantities. The analysis intentionally keeps the historical
model/input conventions except for the explicitly described corrections.
Normalization changes, cleaned SNP inputs and physical-time calibration must
be reflected in regenerated tables/figures rather than mixed with old results.

CPU tests:

```sh
python -m pytest -q analysis/rerun/tests/test_orthogonal.py \
  analysis/genome_wide/tests/test_rerun.py
```
