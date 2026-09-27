# Full-chromosome genome campaign

This protocol changes task arguments and task identities while retaining the
existing, hash-verified `source-v2` inference implementation. It uses one site
block (`core_block_sites=6000000`, `flank_sites=0`) and outer pair chunks of 512.
The native decoder can reduce its internal pair batch to fit free GPU memory;
this does not introduce site boundaries. The physical rates remain
`mu=1.25e-8` and `rho=1e-8`. Gene and lead-window summaries share each complete
chromosome posterior.

The protocol is eligible for deployment only after its native pilot and
cross-host comparison pass. Generating a manifest does not approve it or launch
work. The old `genome_chrN_POP` tasks, outputs, approval gate, manifests and
winner registry remain intact.

## Freeze and launch preparation

Use the original **572-task genome manifest**, not a combined campaign manifest.
The generator preserves its resource requests and scheduling order, replaces
scientific commands with the exact full-chromosome command, and refuses to
overwrite an output manifest:

```sh
python analysis/rerun/full_chromosome_protocol.py \
  --original-manifest ORIGINAL_TASKS_GENOME_JSON \
  --source-manifest FROZEN_SOURCE_V2/SOURCE_MANIFEST.json \
  --output tasks-genome-full.json
```

Freeze and archive the resulting bytes and printed SHA-256. Stage those exact
bytes on each approved host as `control/tasks-genome-full.json`. There are 572
new task IDs, `genome-full_chrN_POP`, covering chromosomes 1–22 and all 26
populations exactly once. Every task requires the separate
`validation/full-chromosome-approved.json` gate. The unchanged frozen worker
recognizes the top-level `validation_gate` field and requires `passed=true`
with the matching frozen source hash before dispatch.

Initially use one worker per allocated GPU. Preserve normal site-specific
Slurm and memory limits. The generator does not infer suitable allocations
from the copied resource requests. Every raw chromosome cache must have at
most 6,000,000 sites; collection checks that bound. Larger retained chromosomes
fail the decoder's zero-flank multi-block check rather than silently using a
finite-context approximation.

The approval JSON must be produced from successful native validation evidence,
not copied from the old blockwise gate. In addition to retained pilot and
cross-host evidence, its required fields are:

- `passed: true` and `protocol: "full_chromosome_v1"`;
- `source_manifest_sha256`: SHA-256 of the unchanged source-v2 manifest;
- `task_manifest_sha256`: SHA-256 of the actual new task-manifest bytes;
- `scientific_settings`: an exact copy of the manifest's settings object;
- `cross_host_parity_validated: true`;
- `validated_hosts`: at least two distinct host labels, including every host
  later supplied for collection.

The collector treats this JSON as an explicit validation attestation. The
approval process must retain and verify the underlying native pilot and
cross-host arrays, tolerances and hardware evidence. It must not mark an
untested protocol as passed.

Only after approval and staging, use the existing source-v2 worker with
`--manifest RUN_ROOT/control/tasks-genome-full.json --kind gpu`. Worker
receipts bind the actual task-manifest hash. The new IDs prevent old completion
or adoption markers from skipping full-chromosome work.

## Separate collection

Synchronize each relevant host's unchanged source-v2 tree, new task manifest,
completion receipts and output artifacts. Use an empty selected-results
directory dedicated to the full-chromosome campaign:

```sh
python analysis/rerun/collect.py \
  --host-root betty=SYNCED_BETTY_ROOT \
  --host-root sesame=SYNCED_SESAME_ROOT \
  --source source-v2 \
  --full-task-manifest tasks-genome-full.json \
  --cross-host-validation full-chromosome-approved.json \
  --output SELECTED_FULL_CHROMOSOME_RESULTS
```

Selection remains incremental and picks the earliest valid completion for each
explicitly mapped chromosome/population. It checks the exact 572-task mapping,
gate, task-manifest bytes on each host, source hashes, executed command,
metadata configuration, physical rates, raw-site bound and output hashes.
Receipts and the immutable winner registry retain the new task-manifest hash.
Old IDs, finite-flank settings and a reused blockwise registry are rejected.
Full-chromosome collection cannot use `--local-validation`; the original
multi-block local-validation rules are unchanged.

Run the bounded regression suite before freezing the orchestration changes:

```sh
python -m pytest analysis/rerun/tests/test_full_chromosome_protocol.py \
  analysis/rerun/tests/test_collect.py analysis/rerun/tests/test_worker.py -q
```
