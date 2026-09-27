Regional iHS plot data — 27 September 2026

GRK2: GIH (Gujarati Indian), BEB (Bengali), IBS (Iberian).
TREML1 / TREM2: IBS, the focal population of the regional case study.
Other populations are not included in these plots.

The plotted window spans the marked gene(s) plus 100,000 bp on either side.
Coordinates are GRCh38, 1-based, with inclusive gene and region endpoints.

CSV columns:
  chromosome: chromosome number.
  position: genomic coordinate in bp.
  population: 1000 Genomes population code.
  source_row: 1-based data-record number in the full chromosome's raw output
              (excluding the header); distinguishes records sharing a position.
  coded_allele_frequency: selscan's frequency under the input VCF's 0/1 coding;
                          not verified ancestral/derived polarity.
  raw_ihs: unnormalized selscan score.
  normalized_ihs: chromosome/population frequency-bin z-score.
  abs_normalized_ihs: absolute value of normalized_ihs, as plotted.

All returned regional rows are preserved, including records at the same
coordinate. Empty normalized values, if any, are unavailable and not plotted.
Variants absent from the raw output are not inserted as zeros. Gene-body
counts refer to score records, consistent with the retained gene summaries.

Normalization is performed on the WHOLE chromosome before region extraction.
It uses 20 coded-allele-frequency bins of width 0.05, closed on the right with
the lowest endpoint included. Each bin requires at least 50 finite raw scores
and a positive finite standard deviation (ddof=0). The dashed plot line is
|normalized iHS| = 2, the descriptive threshold used for gene summaries.

provenance.json records exact gene intervals, source hashes, region bounds,
finite score counts, and per-gene counts. The raw whole-chromosome selscan
outputs and annotation caches are not bundled with these small plot extracts.
The historical runs used selscan v3.0.0 and a constant 1 cM/Mb map; see the
documentation's Methods page for the executed settings.

Extraction and plotting code:
  analysis/orthogonal_v41/scripts/plot_focal_ihs.py

To regenerate images from these CSVs (Python with numpy, pandas, matplotlib):
  python analysis/orthogonal_v41/scripts/plot_focal_ihs.py plot \
    --data docs/_data/focal_ihs --output docs/_static/figures

To re-extract from a repository checkout containing the retained input data,
run through the data host's batch scheduler:
  python analysis/orthogonal_v41/scripts/plot_focal_ihs.py extract \
    --repository /path/to/gamma_smc_cu --output /path/to/output

These files supplement the corrected 2026-09-27 atlas release. They do not
replace or modify its immutable tables.
