# Production genome and lead-bank task manifest

Create this manifest from the frozen production source snapshot, after its
source manifest has been finalized:

```sh
python analysis/rerun/make_genome_tasks.py \
  --source-manifest /runs/run-id/source-v2/SOURCE_MANIFEST.json \
  --source source-v2 \
  --output /runs/run-id/tasks_genome.json
```

The generator submits no jobs and refuses to replace existing task manifests.
It creates all **572** chromosome/population tasks, with stable IDs such as
`genome_chr22_ASW`. Order is chromosomes 22, 11, 6, 12, 20, 2, then the remaining
autosomes in ascending order. Within each chromosome, ASW, YRI and CEU come
first, followed by the remaining populations alphabetically.

Every command invokes `analysis/genome_wide/infer_chromosome.py` with the
strict cache, sample metadata, and lead table. Defaults are core 65,536,
flank 8,192, pair chunk 512, physical mutation rate 1.25e-8 and recombination
rate 1e-8. Lead windows are midpoint ±25 kb, summarized from the same full
chromosome posterior as genes. The default lead path is
`{inputs}/akbari_lead_variants_grch38.tsv`; override `--lead-variants` if the
staged filename differs. All settings are explicit in command arrays and
summarized in `scientific_settings`.

Tasks wait for the chromosome's strict READY cache, gene annotations, sample
file, lead table and `{root}/validation/production-approved.json`. Only create
that approval artifact after the production preflight has passed. The worker
must verify approval content as well as dependency existence when enforcing
that gate; a placeholder file is not scientific approval.

Scientific outputs go to `{output}/results/chrN/POP.{npz,csv,metadata.json}`.
The metadata records the actual cache member and source VCF hashes, gene and
sample hashes, lead-table hash, physical calibration, block/pair settings and
inference source hashes. These identities are consumed by the collector. The
lead bank is embedded in the gene NPZ, so it follows the same selected host
winner. No separate Akbari GPU task is needed.

To merge later with orthogonal and Relate tasks, explicitly select the same
production source directory and manifest:

```sh
python analysis/rerun/make_tasks.py \
  --source-manifest /runs/run-id/source-v2/SOURCE_MANIFEST.json \
  --source source-v2 \
  --merge-task-manifest /runs/run-id/tasks_genome.json \
  --relate-manifest /runs/run-id/outputs/relate/manifest.json \
  --output /runs/run-id/tasks_complete.json
```

Collect production results using `collect.py --source source-v2`. Both hosts
must have identical production source manifests; the cross-host numerical
validation gate must apply to that production snapshot. A separate retained
benchmark snapshot under `source/` does not participate in production
collection.

```sh
python -m pytest -q analysis/rerun/tests/test_make_genome_tasks.py \
  analysis/rerun/tests/test_collect.py
```
