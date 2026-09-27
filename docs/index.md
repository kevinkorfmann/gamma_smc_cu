# Gamma-SMC CUDA

Estimate pairwise coalescence times from phased genomes on NVIDIA GPUs.

## Use the tool

[Install Gamma-SMC CUDA](install.md), then follow the
[VCF walkthrough](usage.md) to infer coalescence times from your own data.
The [Python API](python-api.md) also supports simulated tree sequences,
selected haplotype pairs, and gene or interval summaries.

## Explore the findings

[Browse the 143-locus findings list](findings.md) or download the full atlas
of **19,119 gene rows across 26 populations** from the high-coverage
1000 Genomes panel.

## Latest update · 27 September 2026

- Corrected gene atlas and candidate list, with versioned downloads.
- Full-context pair batching and direct export of dense coalescence estimates.
- Updated accuracy and runtime results: 15.91 seconds for the reported
  63,190-pair V100S benchmark.

[Source code](https://github.com/kevinkorfmann/gamma_smc_cu) ·
[Report an issue](https://github.com/kevinkorfmann/gamma_smc_cu/issues)

```{toctree}
:hidden:
:maxdepth: 1

install
usage
findings
python-api
methods
releases
```
