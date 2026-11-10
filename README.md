# Gamma-SMC CUDA

Pairwise coalescence-time inference from phased genomes on NVIDIA GPUs.
The Python API provides calibrated posterior means, full-context pair batches,
interval summaries and bounded-memory dense export.

[Documentation](https://gamma-smc-cu.readthedocs.io/en/latest/) ·
[Installation](docs/install.md) · [Usage](docs/usage.md) ·
[Latest findings](docs/findings.md) · [Reproducing the analyses](REPRODUCING.md)

The **2026-09-27** findings release contains **19,119 gene rows across 26
populations** and **143 candidate loci**. [Browse the searchable findings list](https://gamma-smc-cu.readthedocs.io/en/latest/findings.html)
or [download the **143-locus CSV**](https://raw.githubusercontent.com/kevinkorfmann/gamma_smc_cu/findings-2026-09-27/findings_database/releases/2026-09-27/stage5_loci.csv).
The list includes each representative gene, all member genes, GRCh38 intervals
and focal populations. [Download the full 19,119-gene atlas](https://raw.githubusercontent.com/kevinkorfmann/gamma_smc_cu/findings-2026-09-27/findings_database/releases/2026-09-27/genome_wide_ranks.csv.gz)
or see [all versioned tables and column definitions](findings_database/releases/2026-09-27/README.md).

Start with the [VCF walkthrough](docs/usage.md) to prepare a chromosome, choose
haplotype pairs, run inference and read the results. The [Python API guide](docs/python-api.md)
covers simulations, streaming and interval summaries.

## Start here

Build from source on Linux x86-64 with an NVIDIA GPU. The checked-in Pixi build
targets A100; the [installation guide](docs/install.md) explains V100 and toolkit
requirements. The public inference API is:

```python
from gamma_smc_cu import infer

# G: complete phased binary matrix, shape (haplotypes, sites).
# positions: strictly increasing base-pair coordinates.
result = infer(G, positions, pairs=[(0, 2)], mu=1.25e-8, rho=1e-8)
mean_generations = result["mean"]
```

Read the [input and calibration requirements](docs/usage.md) before using your
own data, particularly for cropped intervals or nonstandard diploid ordering.
For large outputs use `iter_infer_batches`, `export_dense`, or
`RegionMomentContext` rather than materializing all pairs in host memory.

The implementation builds on [Gamma-SMC](https://github.com/regevs/gamma_smc).
Cite the method and the specific software/data version used in an analysis.

## Reproduce and extend the analyses

[REPRODUCING.md](REPRODUCING.md) maps the corrected full-chromosome workflow,
input versions, validation requirements and analysis scripts. It also indexes
the rank diagnostics, detection/localisation benchmark, selection/BGS experiments
and exploratory profile-matching workflows now included in this repository.

The complete historical recovery package is maintained separately in the private
[analysis-reproduceability repository](https://github.com/kevinkorfmann/analysis-reproduceability).
It preserves frozen source versions, task definitions, environment recipes,
checksum-pinned downloads and final manuscript correction scripts without bulk
datasets. Access to that package is required for the full historical workflow;
the public inference software can be installed and used independently.
