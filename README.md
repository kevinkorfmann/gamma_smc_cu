# gamma_smc_cu

![tests](https://img.shields.io/badge/tests-97%20passed-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-A100%20|%20H100%20|%20B200-76b900)
![license](https://img.shields.io/badge/license-MIT-blue)

**GPU-accelerated pairwise coalescence-time (TMRCA) estimation via the
Gamma-SMC HMM.** A drop-in accuracy-parity replacement for the
reference `gamma_smc` binary (Schweiger & Durbin, *Genome Research*
2023), 12× – 200× faster, that scales to biobank-sized cohorts on a
single GPU.

```python
import gamma_smc_cu

# From an msprime tree sequence:
result = gamma_smc_cu.infer(ts)

# From a phased genotype matrix:
result = gamma_smc_cu.infer(G, positions, mu=1.25e-8, rho=1e-8)

# From a chromosome-scale phased VCF, memory-bounded:
result = gamma_smc_cu.infer_blockwise(G, positions, pairs=my_pairs,
                                  core_block_sites="auto")
```

## Why gamma_smc_cu

Pairwise TMRCA is one of the most information-rich summaries of shared
genealogical history, but it is expensive at scale. For the 1000
Genomes 30× cohort (3,202 individuals, 22 autosomes), a within-
population pairwise scan covers **829,638 pairs and ~2.5 × 10¹²
per-site posterior evaluations** — a scale at which CPU Gamma-SMC takes
days per population.

`gamma_smc_cu` targets this bottleneck. On chromosome 22 for one 1000G
population (178 samples, 63,190 pairs, 402,853 segregating sites):

| Tool | Wall time |
|---|---|
| **`gamma_smc_cu`** (single B200 GPU) | **~24 s** |
| `gamma_smc` CPU reference | ~13 min |
| ASMC (extrapolated from n=1000) | ~43 h |

At genome-wide scale, the full 1000G pairwise scan across 22 autosomes
× 26 populations completes in a few hours of shared-partition B200 time.

## Key features

- **Accuracy parity with Schweiger & Durbin's `gamma_smc`.** Median
  Pearson r of log(TMRCA) against the *msprime* ground truth: **0.876
  (gamma_smc_cu) vs 0.874 (gamma_smc)** across 14 cross-species stdpopsim
  configurations. Matches-or-exceeds the reference on 10 of 14 configs.
- **Gamma-distributed posteriors.** Per-site posterior is parameterised
  as Gamma(α, β) via Schweiger & Durbin's flow-field formulation — no
  time discretisation, continuous-time TMRCA.
- **Memory-bounded blockwise decoder.** Chromosome-scale inputs are
  split into padded site blocks; per-block memory is O(block × n_pairs)
  rather than O(sequence × n_pairs). Handles 200 k+ segregating sites
  for 60 k+ pairs on a single 80 GB GPU.
- **Multi-GPU.** `MultiGPUFlowContext` partitions pair batches across
  all visible CUDA devices.
- **Bitpacked genotype storage.** Haplotype matrix packed at 1 bit/site
  for fast access.
- **Auto-calibration of scaled rates.** `auto_estimate_theta=True`
  infers effective µ and ρ from chromosome-wide heterozygosity,
  matching the upstream binary's behaviour.

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install && pixi run build
```

Requires:
- NVIDIA GPU with compute capability ≥ 8.0 (A40 / A100 / H100 / B200 / RTX 3090+)
- CUDA Toolkit 12.x
- Python ≥ 3.10
- `pixi` (https://pixi.sh)

To reproduce the full manuscript pipeline end-to-end (benchmarks,
genome-wide scan, orthogonal validation, figures, tables, manuscript
compile), see [REPRODUCING.md](REPRODUCING.md).

## Quick start

### From a tree sequence

```python
import msprime, gamma_smc_cu

ts = msprime.sim_ancestry(samples=100, sequence_length=5e6,
                          recombination_rate=1e-8, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=2)

result = gamma_smc_cu.infer(ts)
# result.tmrca: array of shape (n_sites, n_pairs) — posterior-mean TMRCA
# result.positions, result.pairs, etc.
```

### From a phased genotype matrix

```python
import numpy as np
import gamma_smc_cu

# G: (n_haplotypes, n_sites) uint8 array of 0/1 values
# positions: (n_sites,) int64 array of BP coordinates
result = gamma_smc_cu.infer(G, positions, mu=1.25e-8, rho=1e-8)

# Pairwise TMRCA at every segregating site for every pair:
print(result.tmrca.shape)   # (n_segregating_sites, n_pairs)
```

### Restricting to specific pairs

```python
pairs = [(0, 1), (2, 3), (4, 5)]
result = gamma_smc_cu.infer(G, positions, pairs=pairs)
```

### Chromosome-scale inputs

For chromosomes that exceed GPU memory, use the blockwise decoder:

```python
result = gamma_smc_cu.infer_blockwise(
    G, positions,
    pairs=my_pairs,
    core_block_sites="auto",  # picked from free GPU memory
    flank_sites=2048,         # burn-in flanks for HMM boundary effects
    mean_only=True,           # skip full posterior, return mean only
)
```

The blockwise mode processes the sequence in overlapping windows with
256-site forward/backward burn-in flanks on each side, so HMM edge
effects are negligible at the reported central positions.

## Cross-species benchmark

From the 14-configuration stdpopsim suite
(`benchmarks/test_suite_stdpopsim/`):

| Species | Model | Speedup | r(gamma_smc_cu) | r(gamma_smc) |
|---|---|--:|:--:|:--:|
| *Homo sapiens* | OutOfAfrica_3G09 (YRI) | **190.5×** | 0.845 | 0.841 |
| *Pan troglodytes* | BonoboGhost_4K19 (western) | 145.7× | 0.888 | 0.887 |
| *Pongo abelii* | TwoSpecies_2L11 (Bornean) | 114.8× | 0.917 | 0.917 |
| *Canis familiaris* | EarlyWolfAdmixture_6F14 (BSJ) | **178.0×** | 0.877 | 0.873 |
| *Bos taurus* | HolsteinFriesian_1M13 | 100.7× | 0.941 | 0.942 |
| *Arabidopsis thaliana* | SouthMiddleAtlas_1D17 | 41.7× | 0.953 | 0.953 |
| *Drosophila melanogaster* | African3Epoch_1S16 (AFR) | 11.6× | 0.661 | 0.625 |
| *Anopheles gambiae* | GabonAg1000G_1A17 (GAS) | 25.9× | 0.770 | 0.737 |

**Median across all 14 configs: 132× speedup, r = 0.876 (gamma_smc_cu) vs
0.874 (gamma_smc).** See `benchmarks/test_suite_stdpopsim/` for the
full table and reproduction scripts.

## Algorithm overview

The pairwise sequentially Markovian coalescent treats a sequence of
heterozygous/homozygous sites as an HMM observation whose hidden state
is the pairwise coalescence time *t*. Rather than discretising *t* into
bins, Gamma-SMC (Schweiger & Durbin 2023) parameterises the posterior
at each site as a Gamma distribution with two sufficient statistics
(α, β); the site-to-site transition is encoded as a precomputed *flow
field* that maps the current posterior to its post-recombination
value without runtime numerical integration.

`gamma_smc_cu` executes the forward and backward sweeps on the GPU by
assigning one CUDA thread per haplotype pair, with bitpacked genotype
access and a multi-step flow-field cache that amortises the per-site
transition over a precomputed table indexed by inter-site genetic
distance. For chromosome-scale data, the memory-bounded `infer_blockwise`
decoder splits the sequence into padded site blocks, runs forward/back-
ward over each block with burn-in flanks, and stitches the per-block
posteriors back into a single output array.

Two upstream correctness fixes are folded in (both required to reach
reference-parity accuracy):
1. Bilinear interpolation at the upper edge of the flow-field grid
   correctly retains weight on the final cell instead of snapping to
   the previous cell.
2. Per-population segregating-site filter matches the upstream
   `gamma_smc` binary's VCF-parse-time behaviour.

## Main API

| Function | Purpose |
|---|---|
| `gamma_smc_cu.infer(...)` | One-shot pairwise TMRCA for inputs that fit in GPU memory |
| `gamma_smc_cu.infer_blockwise(...)` | Memory-bounded blockwise decoder for chromosome-scale inputs |
| `gamma_smc_cu.CoalescenceEstimator` | Object-oriented estimator with custom discretisation, posterior summaries |
| `gamma_smc_cu.MultiGPUFlowContext` | Partitions pair batches across multiple CUDA devices |
| `gamma_smc_cu.generate_flow_field(...)` | Pre-compute flow fields for non-human / user-supplied demography |
| `gamma_smc_cu.bitpack/unpack` | Fast bit-packed genotype storage helpers |
| `gamma_smc_cu.compute_sfs`, `site_pi`, `pairwise_prefix_scan` | Low-level utilities |

Detailed docstrings on each function.

## System requirements

| Component | Required | Recommended |
|---|---|---|
| GPU | Compute capability ≥ 8.0 (A40) | A100 80 GB / H100 / B200 |
| CUDA Toolkit | 12.0 | 12.4+ |
| GPU memory | 16 GB (chr22, one population) | 80 GB (chromosome-scale blockwise) |
| Python | ≥ 3.10 | 3.12 |
| Host RAM | 32 GB | 128 GB for biobank-scale pair counts |
| OS | Linux x86_64 | Ubuntu 22.04 |

## Pipeline: 1000 Genomes scan

End-to-end scripts for the genome-wide within-population pairwise scan
across 3,202 samples × 26 populations × 22 autosomes live in
`analysis/genome_wide/`:

- `infer_chromosome.py` — per-chromosome, per-population blockwise
  decoder with incremental gene-level accumulators (log-sum,
  running-min, 50-bin histogram).
- `aggregate.py`, `postprocess.py`, `reaggregate_from_npz.py` —
  gene-level geometric-mean aggregation, within-population ranking,
  segmental-duplication masking.
- `slurm_*.sh` — cluster submission scripts (SLURM).

Orthogonal validation pipelines (selscan iHS/nSL, Relate+CLUES2, ASMC,
*cxt*) are in `analysis/orthogonal_v41/` and `analysis/relate_clues/`.
The Akbari 2026 cross-check lives in `analysis/akbari_479_tmrca/`.

## Tests

```bash
pixi run pytest tests/
```

97 unit tests covering the CUDA kernels, flow-field generation,
bitpacking, HMM forward/backward, blockwise stitching, and auto-theta
calibration.

## Citation

A manuscript is in preparation:

> Korfmann K & Mathieson S. *Pairwise coalescence-time inference
> localises a shared West Eurasian sweep haplotype at chr11q13.2*
> (*GRK2*). In preparation, 2026.

If you use `gamma_smc_cu` before the preprint appears, please cite this
repository directly and the reference Gamma-SMC paper:

> Schweiger R & Durbin R. Ultrafast genome-wide inference of pairwise
> coalescence times. *Genome Research* 33(7):1023-1031 (2023).
> doi:10.1101/gr.277665.123

## Related work

- **gamma_smc** (Schweiger & Durbin 2023) — the CPU reference
  implementation that `gamma_smc_cu` re-implements on the GPU.
- **ASMC** (Palamara, Terhorst, Song & Price 2018) — biobank-scale
  pairwise coalescence.
- **PSMC** (Li & Durbin 2011), **MSMC / MSMC2** (Schiffels & Durbin
  2014, 2020) — HMM over het/hom sites for single-pair / multi-pair
  TMRCA.
- **Relate** (Speidel et al. 2019) and **tsinfer / tsdate** (Kelleher
  et al. 2019; Wohns et al. 2022) — ARG-based TMRCA from reconstructed
  tree sequences.
- **cxt** (Korfmann et al. 2026, *PNAS*) — transformer-based pairwise
  TMRCA conditioning on the local SFS.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on top of the Gamma-SMC flow-field formulation of Schweiger &
Durbin (2023). Developed at the University of Pennsylvania (Mathieson
lab, Department of Biology). Benchmark sims use `msprime`
(Baumdicker et al. 2022) and `stdpopsim` (Adrion et al. 2020 /
Lauterbur et al. 2023).
