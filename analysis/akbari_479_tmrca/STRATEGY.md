# tmrca.cu computation strategy for Akbari-2026 lead variants

Per (chromosome, population), the driver in `infer_akbari_windows.py` does:

1. **Load once.** Read the parsed chromosome into memory: the uint8 haplotype × site matrix `G` and the `positions` array from `cache/parsed/chr{N}.npz`. Subset `G` to the haplotypes for the target population.

2. **Calibrate once, globally.** Compute pi_hat (per-individual heterozygosity) on the full chromosome and derive `(eff_mu, eff_rho)` via the same formula gamma_smc uses in its auto-theta mode:
   - `eff_mu  = pi_hat / (4 * Ne)`
   - `eff_rho = pi_hat * (rho/mu) / (4 * Ne)`
   This is the gamma_smc scaling applied at the *chromosome* level, not per-slice.

3. **Loop over Akbari leads on this chromosome.** For each lead variant at position `v`:
   1. **Slice** `G` and `positions` to sites in `[v − 500 kb, v + 500 kb]` (~25 k sites at 1KG density on chr1).
   2. **Decode** with `tmrca_cu.infer_blockwise(G_slice, pos_slice, mu=eff_mu, rho=eff_rho, auto_estimate_theta=False)` for every haplotype pair, in pair-chunks of 2000 to cap peak RAM. The kernel's internal `flank_sites=2048` handles block boundaries inside the slice.
   3. **Aggregate** only the central ±25 kb of decoded sites per pair:
      - per-pair = mean across sites in the window → one TMRCA value per pair
      - accumulate sum / log-sum / log-sq-sum / min / histogram across all pairs
   4. Store geom / arith / min TMRCA + histogram for this lead.

4. **Write** one CSV + NPZ per (chr, pop) with 474 rows in chromosomal order.

## Two window sizes, two jobs

| window | size | role |
|---|---|---|
| **decode slice** | ±500 kb | *HMM context.* Gives the forward-backward decoder enough flanking segregating sites that edge bias at the central ±25 kb is negligible. Wider than LD decay (~50–100 kb of informative flank for recent-TMRCA signal). |
| **aggregation window** | ±25 kb | *Signal of interest.* Localises the reported TMRCA tightly to the Akbari peak so a neighbouring sweep doesn't contaminate it. |

## Two landmine-avoidance choices

**Pre-compute theta from the full chromosome; don't let the kernel re-estimate it per slice.**
The first sanity run used `auto_estimate_theta=True` inside the loop and produced TMRCA values 30× too high (rs11606033 near GRK2: 283 k gens ≈ 8.2 Myr). A ±500 kb slice through a sweep has suppressed pi_hat → auto-theta learns a bogus low effective mu → TMRCA inflates. Freezing `(eff_mu, eff_rho)` to the chromosome-wide value fixed it (same locus now 1,019 gens ≈ 30 kyr).

**Slice per-variant instead of decoding the full chromosome.**
Same per-variant cost either way, but the 474 leads cover a small fraction of the genome. Per-slice saves ~5× total compute vs. decoding whole chromosomes and discarding everything that isn't inside an Akbari window.

## Parallelism

SLURM array over (chr, pop) pairs:
- 22 chromosomes × 26 populations = 572 tasks
- Each task: ~30 s on a dedicated MIG slice, ~3–5 min under 7-way MIG contention on a single B200
- `slurm_infer_akbari_pop.sh` flattens (chr, pop) into the array ID via bash integer division / modulo

## Output schema

`results/chr{N}/{POP}.csv` columns:

```
rsid, chrom, center_pos, window_half_bp,
akbari_X, akbari_S, akbari_posterior,
geom_mean_tmrca, arith_mean_tmrca, min_tmrca,
n_pairs, n_sites
```

`results/chr{N}/{POP}.npz` holds the raw accumulators (lin_sum, log_sum, log_sq_sum, min_lin, min_log, histogram + bin_edges, per-lead metadata) so any alternative summary statistic (quantiles, fraction below threshold, etc.) can be recomputed offline without rerunning inference.
