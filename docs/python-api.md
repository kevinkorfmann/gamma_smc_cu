# Python API

## A small inference

Run this in the installed environment on a GPU. `msprime` is included in the
Pixi environment. This example simulates ten diploid individuals and requests
three explicit haplotype pairs:

```python
import msprime
from gamma_smc_cu import infer

ancestry = msprime.sim_ancestry(
    samples=10, ploidy=2, sequence_length=100_000,
    population_size=10_000, recombination_rate=1e-8, random_seed=42,
)
ts = msprime.sim_mutations(ancestry, rate=1.25e-8,
                          model=msprime.BinaryMutationModel(), random_seed=43)
pairs = [(0, 1), (0, 2), (2, 3)]
result = infer(ts, pairs=pairs, mu=1.25e-8, rho=1e-8)
print(result["mean"].shape)  # (retained segregating sites, 3)
print(result["metadata"])   # rates and conversion to generations
```

Always use the returned `positions`: sites monomorphic in the supplied panel
are removed. `mean` contains posterior mean TMRCA in **generations**, not years.
Multiply by a justified generation interval only when reporting years.

## Use your phased data

Supply a NumPy matrix `G` of shape **(haplotypes, sites)** and a one-dimensional
array of physical base-pair coordinates:

```python
result = infer(G, positions, pairs=[(0, 2), (1, 3)],
               mu=1.25e-8, rho=1e-8)
```

Values must be complete, phased, biallelic `0` or `1`. Missing genotypes,
multiallelic calls, unphased diploid dosages, duplicate positions and unsorted
coordinates are unsupported. Filter deliberately before inference; do not
encode missing values as reference alleles. For VCF input, use the [VCF walkthrough](usage.md).

With default rate estimation, adjacent rows must be the two homologues of
each diploid individual. Coordinates must be finite, nonnegative and strictly
increasing. The calibration denominator is the last retained coordinate plus
one, measured from zero; it is **not callable sequence length**. A cropped
region left at chromosome coordinates therefore needs externally calibrated
rates or a deliberately defined local coordinate/span convention. See
[Time calibration](methods.md#time-calibration) before analyzing a region.

Pairs index rows of `G`, starting at zero. Omitting `pairs` from `infer`
requests every unordered pair, which grows quadratically with panel size.
Use explicit pairs for large panels.

## Stream full-context batches

This keeps the full sequence in the forward/backward calculation while
partitioning pairs. Checkpoint/replay bounds the forward-state allocation;
it does not truncate sequence context.

```python
from contextlib import closing
from gamma_smc_cu import iter_infer_batches

with closing(iter_infer_batches(
    ts, pairs=pairs, pair_batch_size=128, checkpoint_sites=4096,
    gpu_ids=[0], mu=1.25e-8, rho=1e-8,
)) as batches:
    for batch in batches:
        print(batch["pair_offset"], batch["mean"].shape)
        # Consume or save this batch before moving to the next one.
```

Do not collect the iterator into a list when memory is the reason for streaming.
GPU IDs are relative to `CUDA_VISIBLE_DEVICES`; specify `[0, 1]` only if two
devices have been allocated and are visible.

## Export dense means

```python
import json
import numpy as np
from pathlib import Path
from gamma_smc_cu import export_dense

destination = Path("example-dense")  # must not already exist
manifest = export_dense(
    ts, pairs=pairs, output_dir=destination,
    pair_batch_size=128, tile_sites=4096, workers_per_gpu=1,
    gpu_ids=[0], durable=True,
)
assert manifest["complete"]
saved = json.loads((destination / "manifest.json").read_text())
first = np.load(destination / saved["files"][0]["file"], mmap_mode="r")
print(first.shape)
```

Output contains `positions.npy`, `pairs.npy`, `mean_batchNNNNN.npy` and a final
`manifest.json`. Arrays are site-major float32 means; pair order is preserved.
The manifest records batch offsets and calibration. It is written only after
success. An interrupted directory without it is incomplete and is not resumable.
Choose a new destination for another run.

`durable=True` requests file and directory synchronization before return.
The default is buffered output. Disk space still scales with all sites and
pairs even though staging memory is bounded. The exporter does not save
posterior parameters or uncertainty bounds.

## Summarize intervals without dense output

```python
import numpy as np
from gamma_smc_cu import RegionMomentContext

context = RegionMomentContext(ts, mu=1.25e-8, rho=1e-8)
# A physical interval [20,000, 40,000), converted after monomorphic filtering.
left, right = context.positions.searchsorted([20_000, 40_000])
if right - left >= 2:
    moments = context.run(pairs, [(int(left), int(right))])
    geometric_mean = np.exp(moments["region_mean_log"].mean(axis=1))
    print(geometric_mean)  # one geometric mean across sites and supplied pairs
```

Region bounds are
half-open **indices into `context.positions`**, not physical coordinates.
`region_mean` and `region_mean_log` have shape `(regions, pairs)` and use
float64 accumulation. They summarize calibrated posterior means, not posterior
samples of an interval-wide TMRCA.

## Posterior parameters and site blocks

`infer(..., return_posterior=True)` additionally returns `posterior_alpha`
and `posterior_beta`. Convert quantiles using the recorded scale:

```python
from scipy.stats import gamma

posterior = infer(ts, pairs=pairs, return_posterior=True)
scale = posterior["metadata"]["generations_per_coalescent_unit"]
q025 = gamma.ppf(0.025, a=posterior["posterior_alpha"],
                 scale=scale / posterior["posterior_beta"])
```

`mean_only=False` instead adds approximate Wilson–Hilferty bounds as `lower`
and `upper`. These are model posterior summaries, not calibrated selection tests.

`infer_blockwise` splits the site axis with flanking context. When multiple
blocks are used, it approximates full-context inference; increasing the flank
can change results. Prefer the full-context APIs above for exact context,
and validate any site-block approximation on your own data.
