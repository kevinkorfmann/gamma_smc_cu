# Recombination and fixed-reference BGS reruns

After each corrected genome-wide analysis produces `genome_wide_ranks.csv` and
candidate tables, create a new analysis root. For a direct two-host comparison,
use the same validated ranks and the union of the two candidate lists on both
hosts. `stage5_loci.csv` matches the representative-gene atlas;
`stage4_genes.csv` expands it to all qualifying genes. The generator accepts
multiple candidate tables and adds the fixed manuscript examples and controls.
Old Figure-2-only candidates are not silently retained. Case/control population
assignments remain explicit; other genes use the refreshed minimum-rank
population, with YRI added once as a comparison.

```bash
python analysis/rerun/recombination.py prepare \
  --ranks /new/genome_wide_ranks.csv \
  --candidates /new/betty/stage5_loci.csv /new/sesame/stage5_loci.csv \
  --fixed-targets /old/atlas/inputs/targets.tsv \
  --panel /old/atlas/inputs/phase3.panel \
  --bundle /tools/domain-randomized-v1 \
  --vcf-dir /inputs/vcfs \
  --bgs-reference /inputs/fixed_bgs \
  --reuse-atlas /old/atlas \
  --output /new/recombination --shards 8
```

Run this CPU preparation through Slurm on Betty because it hashes the large
local chromosome VCFs. `tasks.json` then contains exact argv lists for GPU
shards, independent BGS annotation, and post-inference summaries/checks.
Use the fastRho Python environment for prepare and inference, so the generated
commands retain the correct interpreter. Execute GPU commands through Slurm
on Betty and directly on Sesame. Each shard uses one visible GPU; the fixed
`cuda:0` refers to the first device exposed to that process.

Retained Sesame inventory:

- Atlas input panel, target manifest and 1.2 GB input cache:
  `/home/kkor/Projects/gamma-smc-recombination-20260924/{inputs,cache}`.
  The corrected-control cache is also checked under `control_update/cache`.
- Model bundle, 188 MB:
  `/home/kkor/funwithgoats/sources/fastrho/domain-randomized-v1`.
- Environment: `/home/kkor/venvs/fastrho/bin/python`, fastRho 0.1.1,
  PyTorch 2.4.0+cu121 in the retained run. Recreate compatible CUDA dependencies
  on Betty; record the actual runtime in each map's JSON.
- Local `chrN.vcf.gz` plus `.tbi`/`.csi` for target chromosomes, using the same
  20220422 GRCh38 phased 3,202-sample source as the retained atlas.
- Both fixed YRI and CEU Buffalo-Kern tracks and their provenance JSON are in
  `manuscript/working/data/recombination_2026/bgs/`. The older Sesame atlas
  contains only YRI, so transfer the current two-population reference folder.

The driver pins fastRho 0.1.1 and verifies the unchanged model and feature-statistic
SHA-256 values embedded in `infer_maps.py`. It copies a historical genotype
cache only when its bytes match a historical input checksum and its complete
extraction/sample/filter fingerprint matches the new task. Changed populations
or regions trigger fresh extraction from the local VCF. VCF hashes, sizes and
modification times are recorded; inference rejects source files changed since
preparation. No old map predictions are copied. Within a new run, completed
maps are skipped only after input/output hashes and model, panel, target,
script and seed metadata match. Split-half maps remove sites fixed within that
half and preserve both haplotypes per person.

Rates retain raw population-scaled rho. The plotted absolute comparison scale
remains rho/(4*10,000); this is a separate declared reference scale from the
corrected Gamma-SMC physical-mutation-rate calibration. Neither auxiliary
fastRho Ne estimates nor map values are substituted into Gamma-SMC in this
descriptive analysis.

The BGS reference consists of published fixed fitted predictions, whose model,
released-summary and exported-map checksums are validated. Gene-level annotations
are recomputed for every refreshed gene and every new target, with exact overlap
weighting, explicit coverage, original masks and no interpolation. The model
is not refitted to the new candidates. If the trusted Dryad fitted object is
available, `analysis/recombination/extract_buffalo_kern.py` can regenerate the
same YRI/CEU fixed predictions before preparation. The older Sesame download
logs record 401/403 failures and do not establish that the source pickle is
present. The already checksummed fixed prediction maps are sufficient for
recomputing annotations.

The separate Barroso reference analysis is preserved in `analysis/bgs/`. To
recompute that sensitivity analysis, transfer the immutable checksummed
`data/bgs/barroso2026` cache and run:

```bash
python analysis/bgs/bgs.py annotate --cache /inputs/barroso2026 \
  --genes /new/genome_wide_ranks.csv --sd /inputs/gene_sd_flags.csv \
  --gene-coordinates one-based-closed --output /new/barroso_annotations
```

These Barroso maps are a different model family from the Buffalo-Kern maps in
the current main figures; keep their outputs and labels separate.

After both hosts validate the maps, re-run `summarize_maps.py` with
`--case-data /new/figure45_case_studies` to calculate associations with refreshed
TMRCA profiles. Without that argument only gene/flank and BGS summaries are
produced. Then use `plot_maps.py` with the refreshed case data. No manuscript
files are edited by the new driver. The private manuscript workflow records the
figure/table dependencies and required numerical refresh.
