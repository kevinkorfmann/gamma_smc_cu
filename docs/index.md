# tmrca.cu

GPU-accelerated pairwise coalescence time estimation.

```{image} ../speed_comparison.png
:alt: Speed comparison
:width: 700px
```

## Install

```bash
git clone https://github.com/kevinkorfmann/tmrca.cu
cd tmrca.cu
pixi install && pixi run build
```

Requires NVIDIA GPU (A100/H100/RTX 3090+).

## Usage

```python
import tmrca_cu

# From a tree sequence
ts = tskit.load("example.trees")
result = tmrca_cu.infer(ts)

# From a genotype matrix
result = tmrca_cu.infer(G, positions, mu=1.25e-8)

# Specific pairs only
result = tmrca_cu.infer(G, positions, mu=1.25e-8, pairs=[(0,1), (2,3)])
```

`G` is an `(n_haplotypes, n_sites)` uint8 matrix. To get it from a VCF:

```python
import cyvcf2
vcf = cyvcf2.VCF("input.vcf.gz")
G = np.array([v.genotype_array()[:, 0] for v in vcf]).T.astype(np.uint8)
```

## How it works

Implements the Gamma-SMC model of
[Schweiger and Durbin (2023)](https://doi.org/10.1101/gr.277665.122)
with a full forward-backward posterior on GPU.
See the [paper](manuscript/main.pdf) for details.

## Citation

```bibtex
@article{korfmann2025tmrcacu,
  title={tmrca.cu: GPU-accelerated pairwise coalescence time estimation},
  author={Korfmann, Kevin and Mathieson, Sara},
  year={2025}
}
```

```{toctree}
:hidden:
:maxdepth: 1

self
```
