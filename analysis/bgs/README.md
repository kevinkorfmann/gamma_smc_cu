# Human background-selection annotations

This workflow downloads recent human B maps, annotates gene intervals, exports
BED masks, and renders exploratory genome-wide and regional plots. It leaves
the input TMRCA estimates and original ranks unchanged.

## Reference maps

The initial source is [Barroso et al. (2026), *Causal inference clarifies the
roles of background selection and mutation rate variation in shaping human
genetic diversity*](https://doi.org/10.64898/2026.06.02.727906), a **preprint**.
The [authors' repository](https://github.com/nwcol/bgs_lmr) is pinned to commit
`1d44ba129d354272c52aee705987cd2e6abd8a97` (September 12, 2026).

- Native GRCh38, autosomes 1–22, YRI predictions at 1-kb bin centers.
- Two models: `split_cds_regulatory` and `split_cds_phastcons`, both with
  `equilibrium_granular_Ne` and `roulette` mutation rates.
- `B` is predicted retained neutral diversity relative to the corresponding
  no-BGS expectation: lower B means stronger predicted background selection.
- The map's recombination input is the hg38 pyrho maps of Spence & Song.
  This does not imply those maps were used for the TMRCA inference.
- These two tracks share data and modeling assumptions. They are annotation
  sensitivity checks, not independent replications or maps for 26 populations.

The downloader verifies each file against the Git blob hash at the pinned
revision and records SHA-256 hashes, URLs, assembly, model, and download date.
Raw references are cached in ignored `data/bgs/barroso2026/`; the approximately
61 MB download is not vendored into Git. Documentation is preserved alongside
the data. A download does not execute upstream code.

An established comparison is [Murphy et al. (2023), eLife](https://doi.org/10.7554/eLife.76065).
Their [released B maps](https://github.com/sellalab/HumanLinkedSelectionMaps/tree/master/Bmaps)
are hg19 and use scaled B/segment-length encoding, requiring an audited interval
liftover for GRCh38. [Buffalo & Kern (2024)](https://doi.org/10.1371/journal.pgen.1011144)
provide another model family, with fitted objects on Dryad. Neither is silently
substituted for the native-GRCh38 source here. Cross-model comparison is still
needed before choosing a principal map for a final analysis.

## Coordinates and coverage

The source contains `chrom,pos,B`; its export script takes `pos` as the center
of a zero-based 1-kb window. The reader validates that positions are ordered,
unique, and congruent to 500 modulo 1000, and that B is finite in [0,1]. For
interval overlap we approximate each center prediction as constant on
`[pos-500,pos+500)`. We never extrapolate through gaps or beyond map coverage.

The gene table used by the current genome-wide inference carries GENCODE
one-based closed coordinates. The CLI requires an explicit coordinate choice;
`--gene-coordinates one-based-closed` changes `[start,end]` to `[start-1,end)`.
Internal intervals and all BED exports are zero-based, half-open.

Gene annotations include overlap-length-weighted mean B, minimum B, map
coverage, and the fraction of **mapped gene bases** below each cutoff. This is
a gene-body annotation, not an average over the SNPs contributing to the
original TMRCA estimator. Map coverage is not equivalent to sequence
callability. Existing callability, SD, and quality masks remain necessary.

## Mask semantics and sensitivity checks

Cutoffs 0.8, 0.9, and 0.95 are exploratory sensitivity choices; none is an
established BGS-free threshold or selected to preserve particular candidates.

- `exclude_B_lt_*.bed.gz`: bins with B strictly below the threshold.
- `keep_B_ge_*.bed.gz`: bins with B at or above the threshold. Intersecting
  with a keep mask also excludes missing map coverage.
- Both sets merge adjacent passing bins without filling gaps.
- A separate **gene-level filter** retains genes with mean B at or above the
  threshold and at least 95% map coverage, excluding SD-flagged genes. This
  does not mean every base of a retained gene passes the site mask.

The output retains original ranks and adds within-population ranks among
retained genes. `mask_sensitivity.csv` reports denominators and original top-1%
retention. It also provides descriptive ranks within B deciles, defined among
non-SD, adequately covered genes with available ranks. These are empirical
stratification summaries, **not p-values, full confounder adjustment, or
evidence ruling out BGS**. Shared B values across populations are not evidence
of independent replication. No TMRCA is divided by B or re-estimated.

## Run on Betty through Slurm

Requires Python with NumPy, pandas, SciPy, and matplotlib. From the software
repository on Betty, put these existing inputs in `analysis/bgs/inputs/`:

- `genome_wide_ranks.csv`
- `sd_flag.csv`
- `case_studies/{GRK2,TREM2}_plot_data.npz`
- `case_studies/{GRK2,TREM2}_gene_track.csv`

The case-study archives are optional for annotation alone; the supplied Slurm
job additionally renders plots and therefore requires them.

```sh
mkdir -p analysis/bgs/logs
sbatch analysis/bgs/slurm_annotate.sh
```

Individual stages:

```sh
python analysis/bgs/bgs.py download
python analysis/bgs/bgs.py annotate \
  --genes analysis/bgs/inputs/genome_wide_ranks.csv \
  --sd analysis/bgs/inputs/sd_flag.csv \
  --gene-coordinates one-based-closed
python analysis/bgs/plot_bgs.py \
  --annotations analysis/bgs/results/gene_bgs_annotations.csv.gz \
  --case-data analysis/bgs/inputs/case_studies \
  --output analysis/bgs/results/figures
```

The compute stages belong in a Slurm allocation. Downloading public references
and the small synthetic interval tests can run locally:

```sh
python -m unittest discover -s analysis/bgs -p 'test_*.py'
```

## Plot integration

Join `gene_bgs_annotations.csv.gz` to a gene plot by `gene_id`, retaining
`b_regulatory_mean`, `b_phastcons_mean`, and the corresponding coverage columns.
For a genomic panel, `read_track()` exposes exact half-open bin boundaries and
B values; keep B on its own 0–1 axis aligned with the genomic coordinates.
For interval plots (such as lead-variant windows), use `summarize_interval()`
on the actual interval rather than assigning the nearest gene's B value.

The renderer produces three exploratory PDF/PNG pairs:

1. Original GRK2 and TREML1/TREM2 regional TMRCA tracks with two aligned B tracks.
   TMRCA lines and interquartile bands are preserved from the archived arrays.
2. Original Manhattan ranks colored by gene mean B, one panel per model.
   Gray crosses identify SD flags or inadequate map coverage.
3. YRI rank versus B and rank distributions by B decile. Box plots show median,
   IQR, and 1.5-IQR whiskers with outliers omitted. These plots are descriptive;
   genes are linked and cannot be treated as independent observations for
   naive significance tests.

Before a final confounder claim, account jointly for local recombination,
gene length, variant count/density, and B; use spatially appropriate uncertainty
or held-out chromosomes; inspect candidate rank stability and known-sweep
recovery. A site-mask sensitivity analysis also requires recomputing the
gene-level summaries on the retained sites; the gene-filter outputs here do
not perform that operation.
