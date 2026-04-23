# tests/betty/ — reproducibility tests that actually run gamma_smc_cu

These tests re-run `gamma_smc_cu.infer_blockwise()` against the real
1000G 30× cache on betty and verify:

1. The production genome-wide scan's per-gene geom-mean TMRCA values
   are intact (paper claim: GRK2 in GIH = 643.8 gen, 21,115 pairs,
   41 sites; SLC24A5 GBR deep sweep; etc.)
2. Fresh inference on a small slice reproduces the **ranking** of
   focal sweep genes within the slice (GRK2 in top 20% of chr11:66-68
   Mb in GIH).
3. Full-chromosome fresh inference matches the production-scan
   per-gene TMRCA values within 1% (bit-level reproducibility).

## Running

```bash
ssh betty
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
pixi run pytest tests/betty/ -v
```

Outside betty, all tests auto-skip (cache dir detection in
`tests/betty/conftest.py`). Override with:

```bash
export GAMMA_SMC_CU_CACHE_DIR=/path/to/parsed
export GAMMA_SMC_CU_RESULTS_DIR=/path/to/results
export GAMMA_SMC_CU_SAMPLES_TXT=/path/to/samples.txt
```

## Test tiers by speed

| Tier | What | Time | GPU? |
|---|---|---|---|
| 1 | Production CSV/NPZ integrity checks | <1 s each | no |
| 2 | Fresh slice inference (500-pair subsample, ±2 Mb) | 30-90 s | yes (MIG slice OK) |
| 3 | Full-chromosome all-pair inference on smallest (chr22, ASW) | 5-10 min | yes (MIG slice OK) |

Run just tier 1:

```bash
pixi run pytest tests/betty/ -v -k tier1
```

Run only fresh-inference tests (skip production checks):

```bash
pixi run pytest tests/betty/ -v -k 'tier2 or tier3'
```

Skip the slowest tier 3:

```bash
pixi run pytest tests/betty/ -v -m 'not slow'
```

## What each test proves

### Tier 1 — `TestTier1ProductionIntegrity`

- `test_tier1_grk2_gih_production_tmrca` — GRK2 in GIH = 643.8 gen at
  21,115 pairs / 41 sites; paper's Methods §Processing numbers.
- `test_tier1_grk2_coordinates` — GRK2 body = chr11:67,266,473–67,286,556
  (GRCh38, as cited in main.tex).
- `test_tier1_slc24a5_low_tmrca_in_gbr` — SLC24A5 GBR < 10,000 gen
  (canonical European skin-pigmentation sweep).
- `test_tier1_grk2_is_within_pop_top_1_percent_in_gih` — GRK2's
  geom-mean TMRCA is in the top 5% of chr11 in GIH (consistent with
  the paper's 0.23% genome-wide rank).
- `test_tier1_pair_count_matches_sample_size` — every production CSV
  row has `n_pairs = n_hap*(n_hap-1)/2`.

### Tier 2 — `TestTier2FreshSliceInference`

- `test_tier2_grk2_tmrca_is_low` — fresh gamma_smc_cu on a ±2 Mb
  slice of chr11 around GRK2 in GIH (seed=42, 500-pair subsample)
  returns per-site geom-mean < 5,000 gen across the GRK2 gene body.
- `test_tier2_grk2_ranks_top_of_slice` — GRK2 ranks in the top 20%
  of annotated genes in the chr11:66-68 Mb slice (fresh inference).

### Tier 3 — `TestTier3FullChromosomeExact`

- `test_tier3_chr22_asw_full_scan_matches_production` — full-chr22
  all-10,878-pair inference in ASW reproduces the production CSV's
  per-gene TMRCA for 5 focal genes within 1% relative error
  (bit-level determinism of gamma_smc_cu).

Marked `@pytest.mark.slow`. Skip with `-m 'not slow'`.
