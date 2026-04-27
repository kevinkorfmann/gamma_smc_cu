# Algorithm

`tmrca.cu` is a CUDA implementation of **Gamma-SMC** (Schweiger and Durbin, 2023),
a moment-matched HMM for pairwise time-to-most-recent-common-ancestor (TMRCA)
inference under the Sequentially Markov Coalescent. This page explains what the
HMM is, what the forward-backward decoding actually computes, and the
moment-matching trick that makes it fast on a GPU.

## The model

For a single pair of haplotypes, the latent state at site $s$ is the TMRCA
$T_s \in (0, \infty)$ in generations. The observation at $s$ is binary:
heterozygous ($X_s = 1$, the two haplotypes differ at this site) or homozygous
($X_s = 0$). The HMM has

- **Prior:** under the standard coalescent with constant $N_e$, the marginal
  prior on $T$ is $T \sim \mathrm{Exp}(1/(2N_e))$, equivalently
  $\mathrm{Gamma}(\alpha=1, \beta=1/(2N_e))$.
- **Emission:** given $T$, the number of mutations on the pair's branches is
  $\mathrm{Pois}(2\mu T \cdot \mathrm{gap}_s)$. For a single site, the
  probability of heterozygosity is $1 - e^{-2\mu T \cdot \mathrm{gap}_s}$. After
  scaling, the per-site emission updates the Gamma posterior with
  $\alpha \leftarrow \alpha + 1$ on a het site and is the identity on a hom site.
- **Transition:** between site $s$ and $s+1$, recombination changes the local
  tree along a span of $\mathrm{gap}_{s+1} \cdot \rho$. The exact SMC' transition
  kernel mixes $T_{s+1}$ over recombination breakpoints and re-coalescence times
  in a non-trivial way. Gamma-SMC's central approximation is to **collapse the
  posterior over $T$ at each site to a Gamma distribution** and let the
  transition act on its two parameters.

## Why Gamma?

Two reasons:

1. **Closed-form for the emission.** The het emission is a Poisson
   observation with rate proportional to $T$. The Gamma is the conjugate prior
   for the Poisson rate, so the post-emission posterior is also Gamma —
   exactly. The kernel can update the state with a single
   `alpha += 1` (in scaled coordinates).

2. **Two parameters per site instead of a histogram.** A standard SMC HMM
   discretizes $T$ into ~32–64 bins and tracks a probability over each, so the
   per-site state is 32–64 floats. Gamma-SMC tracks $(\alpha, \beta)$ — two
   floats — and represents them in `(log10_mean, log10_cv)` coordinates so the
   transition operator can be tabulated on a 51 × 50 grid.

The trade-off is the transition. For a discretized HMM the transition is a
matrix-vector product. For Gamma-SMC, the true Gamma-to-Gamma transition under
SMC' is *not* closed form — it's approximated via a moment-matched **flow field**
(see below). The flow field is precomputed once for fixed $(N_e, \mu, \rho)$ and
applied as a small lookup table during decoding.

## State coordinates

Internally each per-pair state is a 2-vector
$(m, c) = (\log_{10} \mathbb{E}[T], \log_{10} \mathrm{CV}(T))$, where the
expectation and coefficient of variation are over the current Gamma posterior
in *scaled* coalescent time ($T \cdot 2N_e$). Conversion to the natural
$(\alpha, \beta)$ parameterization is

$$
\alpha = 10^{-2c}, \qquad \beta = 10^{-2c - m}
$$

and back the other way

$$
m = \log_{10}\alpha - \log_{10}\beta, \qquad c = -\tfrac{1}{2}\log_{10}\alpha.
$$

The forward kernel works in $(m, c)$ space because (a) the flow field is
defined there and (b) the het emission becomes a single
`alpha += 1` followed by a back-conversion. See
`mc_to_ab()` / `ab_to_mc()` in `src/kernels/gamma_smc_flow.cu`.

## The flow field

The transition operator for a span of $\Delta\mathrm{bp}$ does two things:

1. **Mutation emission integrated over the span:** every base in the span
   contributes a probability of NOT seeing a mutation on the pair's branches.
   In $(\alpha, \beta)$ coordinates this is
   $\beta \mathrel{+}{=} \mathrm{scaled\_mu} \cdot \Delta\mathrm{bp}$.

2. **SMC' recombination over the span:** moves the local tree forward in time,
   effectively reshaping the posterior on $T$.

Schweiger and Durbin show that the second step's effect on the moments can be
captured by a **2D vector field** $(u, v)$ on the $(m, c)$ grid:

$$
\frac{dm}{d\mathrm{step}} = u(m, c), \qquad
\frac{dc}{d\mathrm{step}} = v(m, c).
$$

This field is fitted offline against the true Gamma-SMC moment update for a
particular $\rho$, and stored as a 51 × 50 table of $(u, v)$ floats — the file
`default_flow_field.txt`. For a span of $\Delta\mathrm{bp}$, the iterative
update is

```
for n_iter sub-steps (sized so each step's recombination ≤ MAX_STEP_RHO = 0.1):
    1. mutation emission: β += scaled_mu * (Δbp / n_iter)
    2. recombination:    m += u(m, c) * scaled_rho * (Δbp / n_iter)
                         c += v(m, c) * scaled_rho * (Δbp / n_iter)
    3. clamp (m, c) to grid bounds
```

Bilinear interpolation reads $u$ and $v$ at the current $(m, c)$. This is
implemented in `flow_field_advance()` in `gamma_smc_flow.cu`.

## Forward-backward

For a single pair the standard FB sweep is:

```
forward (s = 0 .. S-1):
    if s > 0: state ← flow_field_advance(state, gap = pos[s] - pos[s-1])
    if X[s]:  state ← apply_het_emission(state)        # alpha += 1
    fwd[s] ← state

backward (s = S-1 .. 0):
    bwd_state := current backward message
    posterior[s] ← combine(fwd[s], bwd_state)
    if X[s]:  bwd_state ← apply_het_emission(bwd_state)
    if s > 0: bwd_state ← flow_field_advance(bwd_state, gap = pos[s] - pos[s-1])
```

`combine()` is the moment-matched product of two Gamma posteriors with the
same scale. In $(\alpha, \beta)$ coordinates and accounting for the marginal
prior:

$$
\alpha_s = \max(\alpha^{\mathrm{fwd}}_s + \alpha^{\mathrm{bwd}}_s - 1, 1),
\quad
\beta_s  = \max(\beta^{\mathrm{fwd}}_s + \beta^{\mathrm{bwd}}_s - 1, \epsilon).
$$

The posterior mean of $T_s$ in real generations is then
$\mathbb{E}[T_s] = (\alpha_s / \beta_s) \cdot 2 N_e$. With CI requested, the
upper and lower bounds use the **Wilson-Hilferty** Gamma quantile approximation:

$$
T_{q} \approx \mathbb{E}[T_s] \cdot
  \left( 1 - \tfrac{1}{9\alpha_s} + z_{q} \sqrt{\tfrac{1}{9\alpha_s}} \right)^3,
$$

with $z_{0.025} = -1.96$ for the lower bound and $z_{0.975} = 1.96$ for the upper.
This is a single FMA-and-cube per site rather than an inverse incomplete-gamma
call, which matters in the inner loop.

## Why this is fast on a GPU

The structural property that makes Gamma-SMC GPU-friendly is **state-per-pair
is constant size and data-independent**. Two floats per site per pair, no
chains of dependent table lookups, no conditional control flow that varies
across pairs. The forward sweep is a textbook stream-compute pattern: 1 thread
= 1 pair, no divergence, no shared state across threads.

A pair-major loop on a CPU does this in single-pair time × number-of-pairs.
Putting one pair on each CUDA thread runs ~50,000 pairs concurrently per A100,
and the only thing the kernel touches per site is:

- two registers for $(m, c)$,
- four bilinear-interpolated values from the flow-field (or, in cached mode,
  from the multi-step cache — see [CUDA optimizations](cuda.md)),
- one bit from the bitpacked genotype matrix.

The memory wall sits at the genotype data and the per-site forward buffer.
[CUDA optimizations](cuda.md) walks through how `tmrca.cu` reduces the
memory traffic so the kernel becomes compute-bound.

## What `infer()` returns

```python
result = tmrca_cu.infer(
    G, positions, pairs=pairs,
    mean_only=False,           # → adds 'lower', 'upper'
    return_posterior=True,     # → adds 'posterior_alpha', 'posterior_beta'
)
```

| key                | shape                  | dtype   | needs                          | meaning                                                                    |
|--------------------|------------------------|---------|--------------------------------|----------------------------------------------------------------------------|
| `mean`             | `(n_sites, n_pairs)`   | float32 | always                         | posterior mean TMRCA in generations                                        |
| `lower`            | `(n_sites, n_pairs)`   | float32 | `mean_only=False`              | 95% lower bound (Wilson-Hilferty)                                          |
| `upper`            | `(n_sites, n_pairs)`   | float32 | `mean_only=False`              | 95% upper bound (Wilson-Hilferty)                                          |
| `posterior_alpha`  | `(n_sites, n_pairs)`   | float32 | `return_posterior=True`        | combined Gamma posterior α (scaled coalescent time)                        |
| `posterior_beta`   | `(n_sites, n_pairs)`   | float32 | `return_posterior=True`        | combined Gamma posterior β (scaled coalescent time)                        |
| `positions`        | `(n_sites,)`           | float64 | always                         | site positions in bp                                                       |
| `pairs`            | `list[(int, int)]`     |         | always                         | haplotype index pairs                                                      |

`mean_only=True` (the default) skips writing `lower`/`upper`, which saves
~40% wall time on the backward pass and 2/3 of the output bytes.

`return_posterior=True` adds the per-site combined Gamma posterior parameters
in **scaled coalescent time** (`T_scaled = T / (2 * Ne)`). The posterior at
site $s$ for a given pair is

$$
T_s \cdot \tfrac{1}{2N_e} \;\sim\; \mathrm{Gamma}(\alpha_s, \beta_s),
$$

so the mean in real generations is `(alpha / beta) * 2 * Ne` (which equals
`mean` to floating-point precision), the variance is
`(alpha / beta**2) * (2 * Ne)**2`, and any quantile $q$ is

```python
from scipy.stats import gamma
T_q = gamma(alpha_s, scale=2*Ne/beta_s).ppf(q)
```

Use this when you need the full distributional shape per site (e.g. for
non-Gaussian uncertainty propagation, custom credible intervals beyond 95%,
or downstream Bayesian analyses).

## References

- Schweiger, R. and Durbin, R. (2023). *Ultrafast genome-wide inference of
  pairwise coalescence times.* Nucleic Acids Research, 51(15):e78.
- Marjoram, P. and Wall, J.D. (2006). *Fast "coalescent" simulation.* BMC
  Genetics 7:16. (SMC' transition.)
- McVean, G.A.T. and Cardin, N.J. (2005). *Approximating the coalescent with
  recombination.* Phil. Trans. R. Soc. B. (SMC.)
