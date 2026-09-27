# Run your VCF

Start with a phased VCF for one chromosome and the [installed tool](install.md).
The example below uses chromosome 22 and 100 diploid samples; replace those
values and filenames with yours. Run from the repository root in the Pixi shell.

## 1. Prepare the chromosome

Add the VCF reader once, then convert your VCF to the tool's array format:

```bash
pixi add cyvcf2
python analysis/rerun/prepare_vcf.py \
  --vcf cohort.chr22.vcf.gz \
  --chr 22 --expected-samples 100 \
  --output inputs/chr22
```

Use a coordinate-sorted, single-chromosome VCF. The converter keeps complete
biallelic SNPs with phased heterozygotes and PASS or unset filters. It removes
repeated positions and records filter counts in `inputs/chr22/READY.json`.
Use a new output directory for each input.

For a multi-chromosome VCF, first select a chromosome with bcftools:

```bash
bcftools view -r 22 -Oz -o cohort.chr22.vcf.gz cohort.vcf.gz
```

This command needs an indexed input. Use `chr22` instead of `22` if that is
how your VCF names the contig. To analyze a population subset, select those
samples before conversion, for example with `bcftools view -S samples.txt`.

## 2. Infer coalescence times

Save this as `run_coalescence.py` and run `python run_coalescence.py` on your GPU:

```python
import json
from pathlib import Path
import numpy as np
from gamma_smc_cu import export_dense

cache = Path("inputs/chr22")
assert (cache / "READY.json").is_file()
G = np.load(cache / "G.npy", mmap_mode="r")
positions = np.load(cache / "positions.npy", mmap_mode="r")
samples = np.load(cache / "sample_ids.npy")

# Each individual has two adjacent haplotype rows: 2*i and 2*i+1.
# This example compares the two haplotypes within every individual.
pairs = [(2*i, 2*i+1) for i in range(len(samples))]

manifest = export_dense(
    G, positions, pairs=pairs, output_dir="results/chr22",
    mu=1.25e-8, rho=1e-8, pair_batch_size=128,
    workers_per_gpu=1, gpu_ids=[0], durable=True,
)
print(manifest["shape"])  # retained sites × requested pairs
```

The example mutation and recombination rates are per base per generation.
Set them for your organism and analysis. Use a full chromosome for this default
workflow; see [calibration](methods.md#time-calibration) for cropped regions.

To compare two different individuals, request their haplotype combinations.
For the first and second VCF samples, use
`pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]`.
To compare all haplotypes in a small panel:

```python
pairs = [(i, j) for i in range(G.shape[0]) for j in range(i)]
```

All-pairs output grows quadratically. For a large panel, start with selected
pairs or use [gene and interval summaries](python-api.md#summarize-intervals-without-dense-output).

## 3. Read the output

```python
output = Path("results/chr22")
manifest = json.loads((output / "manifest.json").read_text())
positions = np.load(output / "positions.npy")
pairs = np.load(output / "pairs.npy")
means = np.load(output / manifest["files"][0]["file"], mmap_mode="r")
print(positions[:5])
print(means[:5, 0])  # first pair, first five sites: TMRCA in generations
```

Each `mean_batch*.npy` stores a batch of pairs, with sites in rows and pairs
in columns. The manifest records the corresponding pair offsets and rate
calibration. The final `manifest.json` marks a completed export.

For posterior intervals, tree-sequence input, multi-GPU batches, and interval
summaries, continue to the [Python API](python-api.md).
