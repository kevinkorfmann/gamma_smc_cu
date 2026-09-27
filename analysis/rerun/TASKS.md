# Complete orthogonal and Relate task manifests

`make_tasks.py` generates immutable worker input; it submits no jobs and runs
no genomic analysis. Its commands use the worker's portable `{root}`, `{source}`,
`{inputs}`, `{tools}`, `{python}` and `{output}` placeholders. Scientific output
lives in distinct task folders under `ROOT/outputs/orthogonal` and
`ROOT/outputs/relate`; the worker's `{output}` holds its attempt log.

First freeze the source with the root orchestration's `SOURCE_MANIFEST.json`.
If Relate is included, create its fresh host-specific manifest after staging
all raw inputs and tools:

```sh
RERUN_ROOT=/runs/new-rerun
python "$RERUN_ROOT/source/analysis/rerun/relate_clues.py" manifest \
  --source-repo "$RERUN_ROOT/source" \
  --output "$RERUN_ROOT/outputs/relate" \
  --relate-root "$RERUN_ROOT/inputs/tools/relate_v1.2.4" \
  --clues-root "$RERUN_ROOT/inputs/tools/CLUES2" \
  --reference-root "$RERUN_ROOT/inputs/Relate_input_files/GRCh38" \
  --vcf-dir "$RERUN_ROOT/inputs/vcfs" \
  --samples "$RERUN_ROOT/inputs/samples.txt" \
  --threads 16 --memory-gb 128

python "$RERUN_ROOT/source/analysis/rerun/make_tasks.py" \
  --source-manifest "$RERUN_ROOT/source/SOURCE_MANIFEST.json" \
  --relate-manifest "$RERUN_ROOT/outputs/relate/manifest.json" \
  --output "$RERUN_ROOT/tasks_orthogonal_relate.json"
```

Omit `--relate-manifest` for orthogonal tasks only. Add `--skip-orthogonal` for
Relate tasks only. `--merge-task-manifest CORE.json` can be repeated to include
existing genome-wide, Akbari or benchmark tasks. Merged manifests must have the
same source directory and source-manifest hash. IDs and output paths must be
unique; dependency cycles fail generation. Relative expected outputs remain
scoped to their individual worker task attempts. Existing manifest files are
never replaced.

The default combined inventory contains **1,469 tasks**: all 1,433 orthogonal
analysis tasks plus five Relate input preparations, five chromosome genealogy
runs, seven focal-population size fits and nineteen CLUES locus runs. The
nineteen loci include all seven TREM2 anchors. Normalization waits for all
22 selscan chromosomes within its population. H12 and neighborhood aggregation
wait for their corresponding 22 chromosome results. ASMC waits for its matching
regional Gamma-SMC/cxt comparison. Relate and population-size stages follow
the freshly generated genealogy dependencies; no historical result marker can
satisfy them.

All orthogonal tasks use `kind=cpu` except the 36 regional tasks, which use
`kind=gpu`. Every Relate/CLUES task uses `kind=cpu` and
`resources.dedicated=true`. **Ordinary CPU workers must exclude dedicated
tasks.** Schedule those explicitly in sufficiently provisioned allocations,
using their stable IDs with the worker's `--only` option. Resource annotations
are requests, not proof of available resources:

- Ordinary orthogonal tasks request 32 GB, or 64 GB for normalization and
  regional comparison. selscan/regional use the configured thread count.
- Relate preparation, chromosome inference and population-size fitting request
  at least 192 GB, allowing headroom above the internal Relate memory limit.
  Population-size fitting requests the manifest's thread count.
- CLUES locus tasks default to 512 GB and 72 hours. Historical memory needs
  ranged from 256 to 900 GB depending on the grid and model size; adjust
  `--relate-locus-memory-gb` to the chosen allocation and observed diagnostics.
  A smaller annotation does not reduce the statistical calculation's memory.

The worker must format absolute expected paths before validation. It must also
check each Relate task's `configuration_manifest_sha256` against the referenced
runtime manifest before execution. The source hash alone does not freeze that
separate run configuration. Completion signals are atomically published
`COMPLETE.json` (orthogonal) and `DONE.json` (Relate).

Use `--targets` to build a different prespecified orthogonal target inventory.
All inference settings are explicit in command arrays; defaults are core
65,536, flank 8,192, H12 window 400/step 50, pair cap 20, mutation rate
1.25e-8 and recombination rate 1e-8. See `ORTHOGONAL.md` and `RELATE_CLUES.md`
for scientific conventions and required host dependencies.

```sh
python -m pytest -q analysis/rerun/tests/test_make_tasks.py
```
