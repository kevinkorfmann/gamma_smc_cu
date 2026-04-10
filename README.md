# tmrca.cu

![tests](https://img.shields.io/badge/tests-97%20passed-brightgreen) ![CUDA](https://img.shields.io/badge/CUDA-A100%20|%20H100-76b900)

GPU-accelerated pairwise coalescence-time estimation via the Gamma-SMC HMM
(Schweiger and Durbin, 2023). At parity with the reference `gamma_smc` binary
on accuracy across 7 species, **25×–190× faster**.

```python
import tmrca_cu

result = tmrca_cu.infer(ts)                        # from a tree sequence
result = tmrca_cu.infer(G, positions, mu=1.25e-8)  # from a genotype matrix
```

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install && pixi run build
```

Requires NVIDIA GPU with compute capability ≥ 8.0 (A40 / A100 / H100 / B200 / RTX 3090+)
and the CUDA toolkit.

## Docs

**[kevinkorfmann.github.io/tmrca.cu](https://kevinkorfmann.github.io/tmrca.cu/)**
