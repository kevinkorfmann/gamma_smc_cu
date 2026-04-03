# tmrca.cu — Coalescence by Exhaustion

![tests](https://img.shields.io/badge/tests-97%20passed-brightgreen) ![CUDA](https://img.shields.io/badge/CUDA-A100%20|%20H100-76b900)

GPU-accelerated pairwise coalescence time estimation.

```python
import tmrca_cu

result = tmrca_cu.infer(ts)  # from a tree sequence
result = tmrca_cu.infer(G, positions, mu=1.25e-8)  # from a genotype matrix
```

![Speed comparison](speed_comparison.png)

![Accuracy](accuracy_hexbin.png)

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install && pixi run build
```

Requires NVIDIA GPU (A100/H100/RTX 3090+) and CUDA toolkit.

## Docs

**[kevinkorfmann.github.io/tmrca.cu](https://kevinkorfmann.github.io/tmrca.cu/)**


[USAGE.md](USAGE.md) | [DEMOGRAPHY.md](DEMOGRAPHY.md) | [demo.ipynb](demo.ipynb)
