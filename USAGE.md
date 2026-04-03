# tmrca.cu — Usage Guide

## Quick start

```python
import numpy as np
from tmrca_cu import FlowContext

# G: haploid genotype matrix (n_haplotypes × n_sites), dtype uint8
# positions: physical positions in bp, dtype float64
ctx = FlowContext(G, positions, Ne=10000, mu=1.25e-8, rho=1e-8,
                  flow_field_path="path/to/default_flow_field.txt")

pairs = [(i, j) for i in range(n) for j in range(i)]
result = ctx.run_fb(pairs, mean_only=True)
tmrca = result["mean"]  # shape [S, n_pairs], in generations
```

## Choosing Ne

The `Ne` parameter controls the output scale: TMRCAs are reported in
generations as `(posterior_mean) × 2 × Ne`. The flow field itself operates
in coalescent units and is independent of Ne — changing Ne only rescales
the output.

### Constant demography

If your data comes from a population with known constant effective size,
pass that Ne directly.

### Variable demography (bottlenecks, growth, structure)

The flow field is precomputed under constant Ne and does **not** need to
be recomputed for non-constant demographies. The flow field encodes
recombination transition dynamics which are robust to demographic
misspecification — our tests show that a constant-Ne flow field produces
well-calibrated TMRCA *correlations* (r = 0.81–0.88 against truth) across
diverse demographic scenarios from stdpopsim.

However, the output *scale* will be biased if you use the wrong Ne.
Under recent population growth, the effective harmonic-mean Ne is
typically smaller than the current census size. To correct this:

**Option A — Estimate Ne from heterozygosity (recommended):**
```python
# Compute average heterozygosity
het = np.mean(G, axis=0)  # allele frequency per site
theta = 2 * np.mean(het * (1 - het)) * n / (n - 1)  # per-site Watterson-like
Ne_harmonic = theta / (4 * mu)
```
Then pass `Ne=Ne_harmonic` to `FlowContext`.

**Option B — Let the data decide:**
If you have ground truth or an external Ne estimate (e.g., from PSMC or
the SFS), use that directly.

The key insight: the *correlation* between estimated and true TMRCAs is
insensitive to Ne (Pearson r is scale-invariant). Only the absolute scale
changes. So getting Ne approximately right is sufficient.

## Flow field

The default flow field (`default_flow_field.txt` from Schweiger et al.)
works well for all tested scenarios, including complex demographic
histories with bottlenecks and population structure.

### Using the default flow field

The flow field shipped with the original gamma_smc tool
(https://github.com/regevs/gamma_smc) is recommended for all use cases.
It is precomputed under constant Ne with the standard coalescent, and
encodes a 51 × 50 grid of (log10 mean, log10 CV) displacement vectors.

### Recomputing the flow field

If you need a custom flow field (e.g., different grid resolution or for
methodological research), the `demographic.py` module can regenerate it:

```python
from tmrca_cu.demographic import generate_flow_field, write_flow_field

# Reproduce Schweiger's default flow field (validated to <0.01% error)
u, v = generate_flow_field()  # ~5 seconds
write_flow_field("my_flow_field.txt", u, v)
```

You can also generate flow fields under demographic models, though our
testing shows this does not improve accuracy:

```python
# Demographic flow field (does NOT improve TMRCA accuracy — see below)
u, v = generate_flow_field(
    Ne_values=np.array([10000, 1000, 50000]),
    epoch_boundaries=np.array([0, 5000, 6000, np.inf]),
    Ne_ref=10000
)
```

### Why demographic flow fields don't help

We extensively tested recomputing the flow field under variable N(t)
using the exact SMC' kernel with epoch-wise incomplete gamma functions.
The demographic flow field differs from the constant-Ne version by ~9%,
but produces **identical** TMRCA accuracy (delta r = 0.0000) on all
tested scenarios, including when using the true N(t) from the simulation.

The reason: the flow field controls the *transition dynamics* during
recombination events. These dynamics are dominated by local posterior
shape, not the global coalescent prior. The demographic bias in TMRCA
estimates is purely a *scale factor* from using the wrong Ne, not a
shape error from the flow field.

## Multi-GPU

For large datasets, distribute work across multiple GPUs:

```python
from tmrca_cu import FlowContext
from concurrent.futures import ThreadPoolExecutor

contexts = []
for gpu_id in range(3):
    _core.set_device(gpu_id)
    contexts.append(FlowContext(G, positions, Ne, mu, rho, ff_path))

# Split pairs across GPUs and run concurrently
def run(ctx, pairs):
    return ctx.run_fb(pairs, mean_only=True)

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(run, contexts[i], pair_chunks[i]) for i in range(3)]
    results = [f.result() for f in futures]
```

Or use the convenience wrapper:

```python
from tmrca_cu.multigpu import MultiGPUFlowContext

ctx = MultiGPUFlowContext(G, positions, Ne, mu, rho, ff_path)
result = ctx.run_fb(pairs)  # automatically distributed
```

Achieves ~2.7× speedup on 3 GPUs for large workloads (499,500 pairs).

## Methods

| Method | Speed | Output | Use case |
|--------|-------|--------|----------|
| `ctx.run_fwd(pairs)` | Fastest | Forward-only mean | Screening, large n |
| `ctx.run_fb(pairs)` | Fast | Forward-backward mean (+ CI) | Best accuracy |
| `ctx.run_fb_summary(pairs)` | Fast | Per-site mean across pairs | Population-level landscape |
