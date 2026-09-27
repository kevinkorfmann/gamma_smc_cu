# Regional full-context replacement and seeded ASMC association

`regional_full_context.py` creates exactly36 regional Gamma-SMC/CXT replacement tasks from a frozen original orthogonal manifest. It executes no scientific computation. The replacements use the same `source-v2/analysis/rerun/orthogonal.py`, targets,20 pairs, physical rates, and reporting windows, with `--core-block-sites6000000 --flank-sites0`. IDs are `orthogonal-full-regional-GENE_POP`; products go to `outputs/orthogonal-full/regional/GENE_POP/`. Original products remain intact.

Generate the portable manifest using a source file whose hash matches the frozen source manifest:

```sh
python analysis/rerun/regional_full_context.py manifest \
  --original-manifest "$CAMPAIGN/control/tasks-orthogonal.json" \
  --source-manifest "$CAMPAIGN/source-v2/SOURCE_MANIFEST.json" \
  --orthogonal-source "$CAMPAIGN/source-v2/analysis/rerun/orthogonal.py" \
  --output tasks-regional-full.json
```

Every task requires the **separate** `validation/regional-full-approved.json` marker. The manifest's top-level `validation_gate` names the same marker so the worker validates its passed status and source identity before dispatch; `requires_gate` retains the association contract. Creating this marker is a coordinator responsibility after actual full-chromosome numerical validation and verification that all relevant raw caches contain≤6,000,000sites. Its minimum association contract is:

```json
{
  "passed": true,
  "protocol": "full_chromosome_v1",
  "source_manifest_sha256": "EXACT_FROZEN_SOURCE_MANIFEST_SHA256",
  "approved_task_manifest_sha256s": ["EXACT_TASKS_REGIONAL_FULL_SHA256"]
}
```

The coordinator should retain and hash-bind the actual supporting validation artifacts in that gate. This utility checks gate approval, protocol, source and exact task-manifest binding; it does not create an approval or infer numerical validation from an existing task file. The gate is independent of any primary genome manifest gate. The task generator refuses modified source identity, missing/extra targets, unexpected original settings, unsafe gate locations, or duplicate task IDs.

## Preserve the ASMC dependency graph

Continue the original36 regional tasks needed by the existing seeded ASMC tasks. Do not redirect those ASMC tasks to the new Gamma output, rewrite their dependency markers, or rerun them solely because Gamma uses a larger context. ASMC consumes the original regional positions and focal pair identities, then decodes genotypes drawn from the immutable cache; it does not consume Gamma or CXT posterior means.

Final association proves that the original and full-context regional generations share:

- Identical source/input fingerprints, including the READY record and its genotype-member hash.
- Identical population haplotypes, positions, focal pairs, gene annotation and reporting window.
- Identical physical calibration and Gamma metadata, pair seed42, and CXT model/subset protocol.
- A cache size within the full-context core bound.

Gamma posterior means may change. CXT output may also differ upon repeat execution, but its model/subset convention and finite shape are checked, and both generations' CXT package/torch provenance are preserved. The selected Gamma and CXT always come from the **same newly completed regional product on the same supplied host**. Seeded ASMC keeps its actual original regional-marker identity in the association registry.

## Validate and materialize a complete local view

Run only on an already-synchronized campaign root containing all required sources, inputs, receipts and products:

```sh
python analysis/rerun/regional_full_context.py associate \
  --root "$CAMPAIGN" \
  --original-manifest "$CAMPAIGN/control/tasks-orthogonal.json" \
  --asmc-manifest "$CAMPAIGN/control/tasks-asmc-regional-v3.json" \
  --full-manifest "$CAMPAIGN/control/tasks-regional-full.json" \
  --output-dir /new/separate/selected-regional-full
```

The output directory must be new and separate from the synchronized root. The utility first invokes the strict existing seeded-ASMC collector: all36 corrected products, all36 original regional dependencies, actual seeded-native gate, exact state grids/subsets, manifests, receipts and artifact hashes must validate. Historical unseeded ASMC is optional; its absence is recorded rather than replaced with stale data. It then checks all36 full-context receipts and panel associations before copying selected regional products.

Selected products are under `regional/GENE_POP/` (new Gamma and CXT together) and `seeded_asmc/asmc/GENE_POP/` (corrected ASMC with its original dependency intact). `association_registry.json` names the original regional, full-context regional and seeded ASMC task IDs, both source generations, task manifests, receipts and actual product hashes. The nested seeded-ASMC registry preserves its historical replacement lineage. `validation.json` is passed only when all36 associations validate. On failure, its errors and incomplete registry remain; no selected regional view is materialized, and no old or failed Gamma product is used as fallback. A nested independently valid seeded-ASMC collection can remain inside an incomplete overall association output; consumers must require the **top-level** passed validation.

This is a same-host collection protocol. It neither mixes old/new outputs silently nor proves hardware parity by comparing filenames. Scientific settings and source generations remain explicit in every selected artifact and registry.
