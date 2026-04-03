# tmrca.cu

GPU-accelerated pairwise coalescence time estimation using the Gamma-SMC model.

![Speed comparison](speed_comparison.png)

## Quick start

```python
from tmrca_cu import FlowContext

ctx = FlowContext(G, positions, Ne=10000, mu=1.25e-8, rho=1e-8,
                  flow_field_path="default_flow_field.txt")

result = ctx.run_fb([(0, 1), (2, 3)], mean_only=True)
tmrca = result["mean"]  # [n_sites, n_pairs]
```

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install && pixi run build
```

Requires NVIDIA GPU (A100/H100/RTX 3090+) and CUDA toolkit.

## Documentation

Full docs: [tmrca-cu.readthedocs.io](https://tmrca-cu.readthedocs.io) (coming soon)

See also: [USAGE.md](USAGE.md) | [DEMOGRAPHY.md](DEMOGRAPHY.md) | [demo.ipynb](demo.ipynb)
