# Collect validated chromosome/population winners

`collect.py` reads already synchronized host roots. It performs no SSH,
submission or inference. Preserve the frozen source manifest (`source-v2/SOURCE_MANIFEST.json` for
production), `state/`, and
the corresponding `outputs/tasks/` layout when synchronizing each host.

```sh
python analysis/rerun/collect.py --source source-v2 \
  --host-root betty=/local/synchronized/betty \
  --host-root sesame=/local/synchronized/sesame \
  --cross-host-validation /local/validation/cross_host.json \
  --output /local/selected
```

The cross-host validation report must contain JSON `"passed": true`; numerical
parity testing is an upstream prerequisite and is not replaced by collection.
If that report records `source_manifest_sha256`, it must match the hosts.
Both selected host source manifests must be identical. `--source source-v2`
selects the production snapshot; the default `source` supports earlier runs.
The earlier benchmark snapshot may remain in `source/` without being mixed
into production. A validation report bound to the benchmark snapshot hash
cannot authorize collection under the production snapshot hash. The collector records the gate's
hash and refuses subsequent collection with a different gate or source.

For each of the 572 chromosome/population combinations, only a successful
`genome_chrN_POP.done.json` is eligible. A failed marker excludes that attempt.
Every artifact listed by the worker must exist with its recorded size and
SHA256. The gene metadata must be complete, in physical-generation units, and
agree with the chromosome/population identity and CSV/NPZ checksums.

Compatibility is checked before copying any new winner. Path-independent
scientific settings include physical rates, block/flank sizes, pair chunking,
other inference options, sample membership hashes and inference source hashes.
The input cache must declare exact one-based int64 positions. Within each
chromosome, original VCF SHA256, all three READY cache-member hashes, cache
shape, gene annotations and explicit sequence length must match. Input schema
and filtering conventions must match throughout the campaign. A conflicting
completed campaign is rejected; the collector does not silently choose one
host's scientific settings.

The earliest recorded successful `completed_epoch` wins; host name breaks an
exact timestamp tie. Host clocks must therefore be synchronized for this
ordering to represent actual completion order. Selected artifacts are copied
into `selected/genome/chrN/POP.{npz,csv,metadata.json}`. The immutable winner
registry preserves the host, original completion record/hash, all worker
artifact hashes, scientific identities and selected-file hashes. Auxiliary
artifacts remain in the synchronized host tree and are inventoried by hash;
this collector only copies the three requested gene summary files.

Run the same command incrementally as more results arrive. Existing selected
files are checksum-verified and are never replaced, even if a later sync
reveals an earlier timestamp on another host. Unregistered files, interrupted
partial copies, changed winners or a changed campaign fail explicitly and
require inspection. Synchronization may be incomplete: missing/corrupted
attempts are recorded as ineligible, and another fully validated host result
can win.

`inspection.tsv` always contains all 572 target rows after successful
inspection. `collection_status.json` reports selected count and
`postprocessing_ready`; it becomes true only at 572 validated winners. Do not
launch production postprocessing before that gate passes:

```sh
python analysis/genome_wide/build_candidates.py \
  --results-dir /local/selected/genome \
  --sd-track /inputs/genomicSuperDups.txt \
  --output-dir /local/new-candidates
```

The candidate aggregator independently rejects mixed scientific settings or
chromosome inputs. Its default requires every chromosome/population result;
partial analysis is only available through its explicit diagnostic override.

```sh
python -m pytest -q analysis/rerun/tests/test_collect.py
```

These tests create small fictional host artifacts. They do not inspect or
collect any production result.
