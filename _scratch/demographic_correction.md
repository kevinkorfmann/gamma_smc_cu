# Demographic Correction for Flow Field Pipeline

## Problem

Both tmrca.cu and the original gamma_smc produce biased TMRCA estimates under
demographic misspecification. The root cause is that the flow field
(default_flow_field.txt) is precomputed under constant Ne. Neither Schweiger's
tool nor ours currently accounts for variable N(t) — this is a limitation of
all existing Gamma-SMC implementations.

Empirical accuracy under misspecification (2Mb, n=20, constant-Ne flow field):

| Demographic model          | tmrca.cu r | Schweiger r |
|---------------------------|-----------|------------|
| Gutenkunst OOA (3-pop)    | 0.814     | 0.773      |
| Tennessen 2-pop           | 0.825     | 0.749      |
| Tennessen African         | 0.882     | 0.863      |

Both methods show the same directional bias: estimates are systematically too
high under demographies with recent population growth, because the constant-Ne
flow field pushes posteriors toward E[T] = 2*Ne at every recombination event.

## Root cause

The flow field encodes: given current posterior Gamma(alpha, beta), after a
recombination event, what is the new posterior? Under constant Ne, the
"recoalesced" lineage draws its new coalescence time from Exp(1) (in
coalescent units). The flow field vectors are the linearized displacement in
(log10_mean, log10_cv) space toward this Exp(1) prior per unit of scaled
recombination rate.

Under variable N(t), the coalescent prior is NOT Exp(1) — it is a mixture of
truncated exponentials determined by the piecewise-constant N(t) history. The
flow field vectors point toward the wrong attractor.

**Schweiger does NOT account for demography.** His generate_canonical_flow_field.cpp
hardcodes Exp(1) as the coalescent prior. The default_flow_field.txt is the only
flow field shipped with the tool, computed once under constant Ne. There is no
option to pass a demographic model. Our proposed fix would be a new contribution
to the Gamma-SMC framework.

## Proposed fix: two-stage pipeline with recomputed flow field

### Stage 1 — Estimate N(t) from a subsample (~1 second on GPU)
- Run 50-100 pairs through the existing constant-Ne flow field pipeline
- Bin the TMRCA posteriors into time epochs and invert coalescent rates
- Output: piecewise-constant N(t) with ~20 epochs

### Stage 2 — Recompute flow field under N(t) + run all pairs
- For each of 2550 grid points (51 mean x 50 CV), compute the new flow
  field vector using the demographic coalescent prior instead of Exp(1)
- This requires solving Schweiger's exact integral (see below) with the
  modified prior
- Write the new flow field to a temp file, load it into FlowContext
- Run all pairs with the corrected flow field — same GPU pipeline, no kernel changes

### The flow field integral (from Schweiger's generate_canonical_flow_field.cpp)

For each grid point (mean, CV) → Gamma(alpha, beta):

1. Compute the "distribution difference PDF": the PDF of coalescence time
   AFTER a recombination event minus the current Gamma PDF. This involves
   confluent hypergeometric 1F1 functions with arbitrary precision.

2. Compute partial derivatives of the Gamma PDF w.r.t. alpha and beta.

3. Solve a least-squares fit: find (du, dv) in log10 coordinates that best
   approximates the distribution change using Gamma parameter derivatives.

Under constant Ne, step 1 uses Exp(1) as the recoalesced prior.
Under variable N(t), step 1 would use the demographic coalescent prior:
  f(t) = lambda(t) * exp(-Lambda(t))
where lambda(t) = 1/(2*N(t)) and Lambda(t) = integral_0^t lambda(s) ds.

For piecewise-constant N(t), this prior has closed-form expressions.

### Implementation plan

Port the core of generate_canonical_flow_field.cpp to Python/scipy:
- Use scipy.special.hyp1f1 instead of arb (2550 grid points, not hot path)
- Use scipy.integrate.quad instead of GSL integration
- Use numpy.linalg.lstsq instead of Eigen SVD
- The modified prior replaces exp(-t) with the piecewise-constant coalescent PDF

This avoids C++ dependency bloat (arb, GSL, Eigen, Boost) and keeps the
correction entirely in Python since it runs once per dataset (~seconds).

## Experimental results (2026-04-02)

### What we tried (and what did NOT work)

1. **Moment-matching flow field from scratch**: Derived flow field vectors
   analytically via derivative of Gamma moments w.r.t. mixing weight.
   Result: values off by 5 orders of magnitude vs Schweiger's original.
   The flow field is NOT a simple moment derivative — it is a least-squares
   fit of the full distribution change to Gamma parameter partials.

2. **Rescaling the default flow field**: Scaled u/v vectors by ratio of
   demographic-to-constant prior moments per grid point. Result: no effect.
   The multi-step cache compounds flow field + mutation emissions iteratively;
   a simple linear scaling of the raw vectors washes out through the
   nonlinear compounding.

3. **Varying Ne parameter**: Tested Ne from 5000 to 20000. The r correlation
   barely changes (0.814 vs 0.815 for Gutenkunst). The bias is always positive
   (estimates too high) regardless of Ne, because the flow field transition
   dynamics push posteriors toward the constant-Ne attractor at every
   recombination event.

4. **Entropy clipping** (from Schweiger's source): Caps differential entropy
   at 1.0 nat by increasing alpha after each flow field step. Tested in cache
   builder (too aggressive, r dropped to 0.73-0.85) and GPU runtime (closer
   to Schweiger but 3-10x slower). Not included — speed loss not justified.

### Key finding

The bias is structural, not a calibration issue. The flow field vectors point
toward the wrong attractor (constant-Ne coalescent prior). No amount of
rescaling, Ne adjustment, or prior calibration fixes this. The only effective
fix is recomputing the flow field under the correct demographic prior.

## Current code

- python/tmrca_cu/demographic.py: DemographicFlowContext class (WIP),
  demographic estimation, flow field I/O infrastructure
- The FlowContext C++ class already supports loading arbitrary flow field
  files — no C++ changes needed for the fix

## Schweiger output format

The binary outputs two float32 planes per chunk: plane 0 = alpha, plane 1 = beta.
Posterior mean TMRCA = (alpha / beta) * 2 * Ne to convert from coalescent units
to generations. The reader.py in the gamma_smc repo confirms this format.


## Final finding (2026-04-02)

### The bias is a scale factor, not a shape error

Testing with the TRUE N(t) from stdpopsim confirms:
- Time-varying recalibration (rescale each site by N(t_est)/Ne_ref):
  reduces median bias from +13k to +5k but slightly hurts r
- Optimal single-Ne recalibration (global scale factor):
  removes all median bias, r unchanged (Pearson r is scale-invariant)
- The optimal Ne is ~5500 (vs 10000), reflecting the harmonic mean of
  N(t) weighted by the TMRCA distribution under recent growth

### What this means

The flow field pipeline produces TMRCA estimates that are:
1. Well-correlated with truth (r = 0.81-0.88) -- this cannot be improved
   by any post-processing or flow field change
2. Systematically scaled by 2*Ne_ref instead of 2*N_harmonic -- this is
   trivially correctable by estimating the harmonic-mean Ne from the data

The demographic correction is NOT a flow field problem. It is a one-line
output rescaling: multiply all TMRCAs by (Ne_harmonic / Ne_ref).

### Implementation

The harmonic-mean Ne can be estimated from the SFS or from the mean
heterozygosity: theta = 4*N_harmonic*mu, so N_harmonic = theta/(4*mu)
where theta is estimated from the data. This is what Schweiger does
when the -m flag is not provided (auto-estimates theta from heterozygosity).
