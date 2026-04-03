# Demographic Correction for Flow Field Pipeline

## Problem

Both tmrca.cu and the original gamma_smc produce biased TMRCA estimates under
demographic misspecification because the flow field is precomputed under
constant Ne. When the true demography has bottlenecks, expansions, or
population splits, the recombination transitions encoded in the flow field
assume equilibrium coalescent rates that don't hold.

Empirical accuracy under misspecification (2Mb, n=20, constant-Ne flow field):

| Demographic model          | tmrca.cu r | Schweiger r |
|---------------------------|-----------|------------|
| Gutenkunst OOA (3-pop)    | 0.814     | 0.773      |
| Tennessen 2-pop           | 0.825     | 0.749      |
| Tennessen African         | 0.882     | 0.863      |

Both methods show the same directional bias (offset from truth in traces and
marginal distributions). tmrca.cu does slightly better because forward-backward
integration partially compensates for biased transitions.

## Root cause

The flow field encodes: given current posterior Gamma(alpha, beta), after a
recombination event with probability p, what is the new posterior? Under
constant Ne, the "recoalesced" prior is exponential(1/2Ne). Under variable
N(t), this prior depends on time -- the coalescent rate lambda(t) = 1/(2N(t))
varies, so the marginal coalescence time distribution is no longer exponential.

## Proposed fix: two-stage pipeline

### Stage 1 -- Cheap demographic estimation (subsample, seconds)
- Run 10-50 pairs through the constant-Ne GPU pipeline
- From the TMRCA posteriors, estimate piecewise-constant N(t):
  bin posterior means into time windows, invert coalescent rate
- Similar to what PSMC does, but leveraging existing GPU speed

### Stage 2 -- Full run with corrected model (all pairs)
- Use estimated N(t) to compute the correct coalescent prior q(t)
- Run all pairs with this prior

## Where the correction enters -- three options

### Option 1: Prior correction only (easiest, ~80% of the fix)
- Keep the constant-Ne flow field
- Replace the exponential coalescent prior with N(t)-derived prior
- The flow field handles recombination mechanics (don't change much with demography)
- Main bias is in the prior mixed in during recombination transitions
- Implementation: modify the initial (mean, CV) and the recombination "reset" target
  in the cache builder and kernel initialization
- Infrastructure exists: compute_coalescent_prior already takes piecewise-constant N(t)
  in the HMM path; the flow field path needs a similar mechanism

### Option 2: Recompute flow field under N(t) (harder)
- Schweiger's generate_canonical_flow_field.cpp solves SMC' transitions under constant Ne
- Under variable N(t), coalescent rate in "recombine and recoalesce" depends on time
- Need to solve a different ODE for each (mean, CV) grid point
- Expensive precomputation but only done once per demography

### Option 3: Iterative EM (most principled)
- Alternate between estimating TMRCAs and re-estimating N(t), like PSMC
- adaptive_prior_infer EM loop already exists in the codebase
- Extend it with a richer demographic model (piecewise-constant N(t))
  instead of just a single prior vector

## Investigation notes

### Entropy clipping (tested, not included)
Schweiger's gamma_smc applies entropy clipping after each flow field step:
if differential entropy of the Gamma posterior > 1.0 nat, it increases alpha
(concentrates the distribution) until H <= 1. Uses precomputed lookup table.

Tested in tmrca.cu:
- Cache-only clipping: accuracy dropped (r=0.73-0.85), too aggressive
- Runtime kernel clipping: closer to Schweiger but 3-10x slower (lgamma/digamma in hot path)
- Decision: not included. Speed loss not justified; tmrca.cu already outperforms Schweiger

### Schweiger output format
The binary outputs two float32 planes per chunk: plane 0 = alpha, plane 1 = beta.
Posterior mean TMRCA = (alpha / beta) * 2 * Ne to convert from coalescent units
to generations. The reader.py in the gamma_smc repo confirms this format.
