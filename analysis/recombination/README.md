# fastRho maps at study-highlighted genes

This analysis supplements the Gamma-SMC study with population-specific,
LD-derived recombination maps. `prepare_targets.py` extracts the named genes
from the revision main text, SI, and Tables 1–2, and includes all 165 highlighted
Figure 2 loci. Aliases are resolved explicitly; the pseudogene TREML3P is
covered by the TREM-region map but has no protein-coding rank-table entry.

## Reproduce

```bash
python prepare_targets.py --revision /path/to/manuscript/working --output inputs
cp /path/to/manuscript/working/data/figure45_case_studies/1kgp_phase3.panel inputs/phase3.panel
python infer_maps.py --root "$PWD" --bundle /path/to/domain-randomized-v1 --genes GRK2 TREM2 --split-check
python infer_maps.py --root "$PWD" --bundle /path/to/domain-randomized-v1
```

Inference needs the [fastRho 0.1.1](https://github.com/kevinkorfmann/fastrho)
GPU environment, the checksummed
[domain-randomized-v1 model](https://github.com/kevinkorfmann/fastrho-models/releases/tag/domain-randomized-v1),
cyvcf2, bcftools, pandas and certifi. Plotting needs NumPy, pandas, SciPy and Matplotlib.
`python -m unittest discover -s . -p 'test_*.py'` checks interval aggregation.
Large caches, maps and run logs are ignored by Git; revision summary data and
figures are archived separately in the private manuscript.

## Inputs and interpretation

* Coordinates: GRCh38, 0-based half-open in output TSV/BED files. VCF positions
  are converted from 1-based exactly once. GENCODE intervals are converted
  from the study's 1-based start/end to half-open intervals.
* Cohort: Phase-3 sample membership, selected from the NYGC 3,202-sample
  high-coverage phased GRCh38 VCF. Use each gene's study case population or
  minimum-rank population, and YRI. No population pooling. These are new
  population-wide estimates, not estimates from the 20 pairs plotted for TMRCA.
* Filters: PASS/unfiltered, biallelic ACGT SNPs; drop missing and unphased
  heterozygous calls per population, fixed sites, and duplicate positions.
  No MAF threshold, imputation or SNP thinning. All retained sample IDs and
  filter counts are recorded per map.
* Context: display gene ±500 kb (existing figure windows for GRK2/TREM2),
  infer with another 500 kb on either side. Positions are translated to
  local extraction coordinates for inference, then restored.
* Model: one unchanged general-purpose checkpoint and companion feature
  statistics, phased view, mutation rate 1.25e-8. No retraining.
* Scale: preserve population-scaled rho/bp. The absolute rate is
  rho/(4 × 10,000), using the same Ne as the Gamma-SMC analysis. Rates in
  cM/Mb multiply that conditional rate by 1e8. Auxiliary model Ne is recorded
  but is not used to rescale each locus. A 1 cM/Mb reference corresponds to
  the original Gamma-SMC constant input rate, 1e-8/bp/generation.
* Reporting: exact adjacent-SNP intervals plus overlap-weighted 10-kb and
  50-kb windows. Uncovered portions are recorded, never filled with zero or
  extrapolated. Per-interval model bounds are retained, but averages of them
  are not called calibrated confidence intervals for window means.
* Repeatability: for the two main genes, split diploid samples deterministically
  into two disjoint groups, retaining both haplotypes per person. These runs
  diagnose sample sensitivity; they are not independent historical evidence.

Selection, demography, relatedness, ascertainment and structural variation can
influence an LD-derived map. Spatial correspondence with Gamma-SMC output is
descriptive: this analysis alone cannot establish that recombination caused a
TMRCA signal, or establish robustness to replacing Gamma-SMC's input map.

## Buffalo–Kern background selection

The aligned BGS track is from [Buffalo and Kern (2024)](https://doi.org/10.1371/journal.pgen.1011144),
using the checksummed `cadd6__decode__altgrid.pkl` from
[Dryad version 275846](https://datadryad.org/dataset/doi:10.5061/dryad.qnk98sfnv).
`extract_buffalo_kern.py` evaluates the initial YRI and CEU sparse fits at 100 kb,
with the published fitted parameters and the authors' `predict_simplex`
formula (neutral diversity normalized to one). It reads the trusted-hash
pickle with a restricted reader and inert bgspy objects; no bgspy code is
imported. The deposited object is the alternative-grid fit. The separate
1-Mb `cadd6_summary.tsv` was generated from `cadd6__decode` in the authors'
`revisions.ipynb`, so its values are not an equality check for this fit.
The output provenance records this distinction and the numerical comparison.

```bash
python extract_buffalo_kern.py --model /path/to/cadd6__decode__altgrid.pkl \
  --summary /path/to/cadd6_summary.tsv --output bgs --populations YRI CEU
python summarize_maps.py --root . --ranks /path/to/per_gene_tmrca_and_ranks.csv \
  --case-data /path/to/revision/data/figure45_case_studies
python plot_maps.py --root . --case-data /path/to/revision/data/figure45_case_studies
python validate_run.py --root .
python archive_revision.py --root . --revision /path/to/revision
```

B′ is the retained fraction of neutral diversity; low values predict stronger
BGS. Original GRCh38 bin boundaries and masks are retained, with no smoothing,
interpolation or gap filling. Gene overlaps do not create finer resolution:
TREML1 and TREM2 share one 100-kb bin. The case-study figures compare CEU and YRI reference fits; neither is a fit
to the focal GIH or IBS sample. Gene summaries and the standalone atlas retain
the YRI annotation. These fits are distinct from the Barroso et al. 2026 maps
in `analysis/bgs`.
Among the 242 highlighted genes, 205 have complete B′ body coverage, six partial
coverage and 31 no body coverage. Missing B′ is never interpreted as weak BGS.
BGS, demography and positive selection, including combinations, remain possible
explanations for the TMRCA patterns; these tracks do not perform model selection.

## Run and presentation record

The completed run is on `ssh sesame`, under
`/home/kkor/Projects/gamma-smc-recombination-20260924`. Eight resumable shards
produced all 480 requested maps, and eight additional focal split maps passed
the same output checks. Five control-population assignments were corrected in
a separate `control_update` run. The initial and final manifests are retained;
the final manifest selects the intended case-study/control population and YRI.
Superseded maps remain on Sesame but are excluded from the final atlas/archive.

The VCF source is the 20220422 3,202-sample phased SNV/INDEL/SV release; Phase-3
membership is applied explicitly. Input genotypes and BCF caches remain on
Sesame. The archive includes all output maps locally; large NPZ/BED files are
Git-ignored, while window means, metadata, checksums and figure inputs can be
versioned. The focal and full-atlas figures have no plot titles or subtitles;
their descriptions and interpretation are in `figure_legends.md` and manuscript
captions. Panel letters, axis labels and gene/population legend keys remain.
