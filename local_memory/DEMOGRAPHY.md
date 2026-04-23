# Demographic Robustness and Limitations

## Summary

The constant-Ne flow field produces well-calibrated TMRCA *correlations*
(r = 0.81–0.88) under diverse demographic scenarios without modification.
The only bias is a scale factor from using the wrong Ne, fixable by
estimating the harmonic-mean Ne from heterozygosity.

This document explains why, and what would be needed for a fully rigorous
demographic treatment.

## What we tested

Using three demographic models from stdpopsim (Gutenkunst out-of-Africa,
Tennessen two-population, Tennessen African) with constant-Ne flow field:

| Approach | r change | Speed cost | Result |
|----------|----------|------------|--------|
| Recompute flow field under true N(t) | +0.000 | +10s setup | No improvement |
| Entropy clipping (Schweiger's regularizer) | -0.03 to -0.05 | 3-10x slower | Hurts accuracy |
| Time-varying output recalibration | -0.01 to -0.03 | none | Slightly worse |
| Single optimal Ne (scale factor) | +0.000 | none | Removes bias, r unchanged |

## Why the flow field doesn't need demographic correction

The flow field encodes the linearized change in Gamma posterior parameters
per unit of recombination. This change is dominated by the *local posterior
shape* (how peaked or diffuse the current estimate is), not by the global
coalescent prior. Under variable N(t), the posterior shape at any given site
is determined primarily by nearby mutations, not by the recombination
transition dynamics.

We verified this by regenerating the flow field under the exact SMC' kernel
with piecewise-constant N(t), using epoch-wise incomplete gamma functions.
The resulting flow field differs from constant-Ne by ~9%, but this difference
is too small to affect the posterior after passing through the multi-step
cache compounding and forward-backward combination.

## What the bias actually is

The output is `(alpha/beta) * 2 * Ne`, where alpha/beta is the posterior
mean in coalescent units. Under constant Ne, `2 * Ne` correctly converts to
generations. Under variable N(t), the correct conversion depends on *when*
coalescence occurs. The effective conversion factor is `2 * N_harmonic`,
where N_harmonic is the harmonic mean of N(t) weighted by the coalescent
time distribution.

For populations with recent growth, N_harmonic < N_current, so using
N_current as Ne systematically overestimates TMRCAs. The fix:

```python
# Estimate N_harmonic from heterozygosity
theta_per_site = mean_heterozygosity  # average across diploid pairs
Ne_harmonic = theta_per_site / (4 * mu)

# Use this Ne for the pipeline
ctx = FlowContext(G, positions, Ne=Ne_harmonic, mu=mu, rho=rho, ...)
```

## What full rigor would require (not implemented)

### 1. Non-equilibrium initial prior

The forward pass initializes at (mean, cv) = (0, 0), which implicitly
assumes the equilibrium coalescent prior. Under variable N(t), the
correct initial Gamma posterior should moment-match the demographic
coalescent time distribution. This is a single computation per run
(derive alpha_0, beta_0 from the first two moments of the demographic
prior) and could improve estimates at the edges of chromosomes where the
prior matters most. Expected impact: small (<1% in r).

### 2. Time-dependent rate scaling

The scaled parameters `theta = 4*Ne*mu` and `rho_scaled = 4*Ne*rho` are
computed once and used everywhere. Under variable N(t), the effective
rates depend on time: at coalescent time t, the relevant Ne is N(t).
This means the flow field step size should vary with the current posterior
mean — sites where the estimated TMRCA is recent (small N) should use
different scaling than sites where it is ancient (large N).

Implementing this would require parameterizing the flow field cache by
the current posterior mean, breaking the precomputation model.
Alternatively, the iterative (non-cached) kernel could adjust rates
per-step, at the cost of slower execution. Expected impact: moderate
(1-3% in r), significant engineering.

### 3. Non-parametric posterior

The Gamma family is unimodal and right-skewed. Under complex demography
(e.g., a bottleneck creating bimodal coalescent time distributions), the
true posterior can be multimodal. No Gamma moment-match captures this.

The rigorous fix: use the discrete-state HMM with K time bins, which
can represent arbitrary posterior shapes. This already exists in the
codebase (`hmm_forward_backward_gpu`) and supports demographic priors
via `compute_coalescent_prior`. However, it scales as O(K) per site per
pair instead of O(1), making it ~32-128x slower per pair than the flow
field approach.

For demographic robustness where the Gamma approximation fails, using
the K=32 HMM with a demographic prior is the correct approach. It is
still much faster than Schweiger's CPU tool.

### 4. Joint EM: TMRCA + demographic inference

The most principled approach: iterate between estimating TMRCAs given
N(t) and estimating N(t) given TMRCAs. This is what PSMC does for a
single pair. Extending it to all pairs with GPU acceleration would
enable demographic inference from the pairwise composite likelihood
(related to MSMC2).

The `adaptive_prior_infer` function already implements EM over the
coalescent prior. Extending it to infer a piecewise-constant N(t)
instead of a single prior vector would require:
- Parameterizing the prior as N(t) with M epochs
- Computing the coalescent prior q(k) from N(t) at each EM step
- Gradient or moment-based updates for the N(t) parameters

This is a research contribution beyond the scope of the current tool.
Expected impact: would jointly solve scale and shape bias, at the cost
of multiple forward-backward passes (~5-10 iterations).

## Flow field regeneration

The `demographic.py` module can regenerate Schweiger's flow field from
scratch using `scipy.special.gammainc` (no arb/GSL/Boost dependencies):

```python
from gamma_smc_cu.demographic import generate_flow_field, write_flow_field

# Constant-Ne (matches Schweiger's default to <0.01% error, ~5 seconds)
u, v = generate_flow_field()
write_flow_field("my_flow_field.txt", u, v)

# Demographic (validated but does not improve accuracy)
u, v = generate_flow_field(
    Ne_values=np.array([10000, 1000, 50000]),
    epoch_boundaries=np.array([0, 5000, 6000, np.inf]),
    Ne_ref=10000
)
```

The constant-Ne regeneration is useful for:
- Verifying the flow field (validated against Schweiger's to 0.005%)
- Generating flow fields with different grid resolutions
- Methodological research on the Gamma-SMC model
