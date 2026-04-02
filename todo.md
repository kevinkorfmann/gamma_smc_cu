# tmrca.cu — TODO

## Figures
- [ ] Restyle fig6_demographics to Nature style (currently plain matplotlib, others are Nature-formatted)
- [ ] fig3 boxplot uses hardcoded/simulated data in nature_figures.py — replace with real per-pair accuracy data
- [ ] Add 10Mb data to Nature-style figures (fig1, fig2 currently only show 1Mb + 5Mb)

## Manuscript text
- [ ] Table 1: 10Mb rows use estimated values — run actual benchmarks and fill in real numbers
- [ ] Table 1: add missing rows (n=20, n=1000 for 5Mb and 10Mb)
- [ ] Grep for broken figure/table refs (fig:speedup, fig:accuracy label mismatches)
- [ ] Discussion: verify throughput claim "approximately $2 \times 10^6$ pairs per second"
- [ ] S1_Text: run bibtex pass (currently no .bbl for SI)
- [ ] S1_Text: audit for references to old HMM-path APIs/kernels that no longer match flow field code

## Bibliography
- [ ] Verify Schweiger2023 entry matches actual publication (journal, year, doi)
- [ ] Verify Adrion2020 and Gutenkunst2009 are in references.bib

## Code / benchmarks
- [ ] Run full benchmark at 10Mb and 100Mb with real Schweiger times to fill Table 1
- [ ] Clean up src/kernels/gamma_smc.cu.bak (delete or gitignore)
- [ ] Consolidate make_figures.py and nature_figures.py into one script
- [ ] Implement multi-GPU support (3x more available, mentioned as future work)

## Before submission
- [ ] Replace GitHub URL placeholder in Methods section
- [ ] Add ORCID for authors
- [ ] Write cover letter
- [ ] Check PLOS Comp Bio word limits and formatting checklist
- [ ] Proofread notation consistency (Gamma-SMC vs gamma_smc vs Gamma SMC)
