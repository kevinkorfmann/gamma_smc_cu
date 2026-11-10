# Reproducing the analyses

The corrected public findings release is **2026-09-27**: 19,119 gene rows
across 26 populations and 143 candidate loci. Its tables and source hashes are
preserved under
[`findings_database/releases/2026-09-27/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/findings_database/releases/2026-09-27).
The later research workflows listed below have separate inputs, protocols and
validation; publishing their source does not replace that findings release.

This guide supersedes the earlier instructions for the historical 165-locus
analysis. Those instructions remain in Git history. In particular, historical
finite-flank runs, old CLUES fits and a missing local download script are not
an execution recipe for the corrected analysis.

## Recover the original analysis without the original machines

The separate private
[analysis-reproduceability repository](https://github.com/kevinkorfmann/analysis-reproduceability)
contains the collected Betty, Sesame and local scripts, frozen source trees,
exact task definitions, environment recipes, small fixed analysis definitions,
checksum-pinned downloaders, figure/table adapters and subsequent corrections.
Access to that private repository is required for this complete historical
package; it is not required to install or use the public inference software.

```bash
git clone https://github.com/kevinkorfmann/analysis-reproduceability.git
cd analysis-reproduceability
python3 tools/verify.py
python3 tools/create_run.py --destination /runs/reproduction
python3 tools/download_inputs.py --destination /runs/reproduction/inputs --dry-run
python3 tools/download_resources.py --destination /runs/reproduction/inputs --dry-run
```

Follow its
[RUNBOOK.md](https://github.com/kevinkorfmann/analysis-reproduceability/blob/main/RUNBOOK.md)
for the actual build, download and execution commands. The recovery directory
is assembled offline using Python's standard library; it does not require SSH,
submit jobs, or install old completion/validation markers.

The preserved production inference source has 376 manifest-bound files;
matched runtime and corrected ASMC use a separate 380-file source snapshot.
The current public checkout includes later orchestration and API changes.
Retain the frozen source identities when reproducing those original runs.

## Inputs and compute

The input release is the phased high-coverage 1000 Genomes
`20220422_3202_phased_SNV_INDEL_SV` panel, with 3,202 samples. The 22 autosomal
VCFs and their indexes total about **29.8 GB**; regenerated strict chromosome
caches require about **384 GB**, before references and analysis outputs.
The recovery download manifest records the exact sizes and SHA-256 hashes.
Do not substitute a different release because its filenames look similar.

The strict cache builder is
[`analysis/rerun/prepare_vcf.py`](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/prepare_vcf.py).
Its
[format and filtering rules](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/PREPARE_VCF.md)
cover biallelic SNPs, complete phasing, duplicate positions and manifest-bound
cache members. The corrected cache uses per-chromosome directories containing
arrays and a validated READY record, rather than the old single NPZ layout.

Use Linux with compatible NVIDIA hardware and CUDA for GPU inference. Run
builds and analyses through the site's scheduler; Betty uses Slurm. Adapt
allocation names and filesystem paths to a replacement machine while preserving
scientific parameters, seeds, sample ordering and source hashes. Storage,
concurrency and hardware determine runtime; there is no verified universal
one-day execution estimate.

References and model weights are downloaded separately. The recovery package
pins cxt, fastRho, Relate references and the segmental-duplication track, and
verifies the original Relate 1.2.4 programs. Optional Buffalo–Kern fitted objects
used by B-prime refitting returned HTTP 403 from Dryad during the November 2026
check; the recovery runbook records exact file IDs, hashes and a manual-download
fallback. Frozen compact BGS-map parameters support ordinary map annotations.

## Corrected workflow

| Stage | Implementation and required protocol |
| --- | --- |
| Strict inputs | `analysis/rerun/prepare_vcf.py`; retain the exact population, gene and lifted Akbari target definitions. |
| Genome-wide inference | 572 chromosome/population tasks using full-chromosome context, core 6,000,000 sites, zero flanks, pair chunks 512, `mu=1.25e-8` and `rho=1e-8`. |
| Collection | Use the current full-chromosome-aware `analysis/rerun/collect.py`, the exact full task manifest and fresh validation evidence. The collector inside the older frozen source predates this protocol. |
| Candidate tables | `analysis/genome_wide/build_candidates.py` and `lead_summaries.py` operate on complete selected full-chromosome results. Preserve ranking, segmental-duplication and population conventions. |
| Orthogonal comparisons | `analysis/rerun/orthogonal.py`, full-context regional replacement tasks and corrected seeded ASMC. Preserve the original regional dependencies required by ASMC. |
| Relate and CLUES2 | `analysis/rerun/relate_clues.py` describes the staged genealogy workflow. The final 13 fits additionally require the later focal-tree, population-order, demographic and marker corrections preserved in the recovery repository. |
| Recombination and BGS | `analysis/rerun/recombination.py` and `analysis/bgs/`; retain the fixed targets, model weights, reference-map versions and physical calibration. |
| Benchmarks | Cross-species accuracy, matched runtime and dense-export throughput have distinct protocols and source generations. |
| Figures and tables | Regenerate from newly verified results using the recovery package's manuscript adapters and final correction scripts. Manuscript sources remain in their separate private repository. |

The
[full-chromosome protocol](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/FULL_CHROMOSOME_PROTOCOL.md)
and
[regional/ASMC association protocol](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/REGIONAL_FULL_CONTEXT.md)
specify validation gates, manifest identity and collection. Generate fresh native
pilot evidence before production. Historical approval files are provenance, not
proof that a replacement environment passed. The original multi-host protocol
requires cross-host parity evidence; running on a single replacement host does
not silently waive that requirement.

The
[recombination/BGS guide](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/RECOMBINATION_BGS.md)
and
[Relate/CLUES guide](https://github.com/kevinkorfmann/gamma_smc_cu/blob/main/analysis/rerun/RELATE_CLUES.md)
describe their input fingerprints and resource requirements. The original
19-locus Relate task manifest alone does not reproduce the final 13 corrected
CLUES fits. Follow the correction chain and parameter inventory in the recovery
runbook before generating manuscript summaries.

## Analysis and benchmark source index

| Directory | Scope |
| --- | --- |
| [`benchmarks/test_suite_stdpopsim/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/test_suite_stdpopsim) | Cross-species accuracy and CPU/GPU parity. |
| [`benchmarks/pairwise_scaling/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/pairwise_scaling) | Matched runtime, pair scaling and the separate dense-export benchmark. |
| [`analysis/rank_diagnostics/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/analysis/rank_diagnostics) | Covariate dependence, retained-site audits and iHS/XP-EHH comparisons. |
| [`benchmarks/detection_localisation/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/detection_localisation) | Fixed 1,000-replicate detection/localisation experiment; the production Gamma score uses the CPU reference. Includes later source-data audits and figure scripts. |
| [`benchmarks/selection_robustness/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/selection_robustness) | SLiM selection/BGS experiments, B-prime refitting, stronger-BGS follow-ups and spatial calibration/plotting. Individual experiment gates remain in force. |
| [`benchmarks/autoresearch_profiles/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/autoresearch_profiles) | Exploratory profile matching with fixed scoring and independent simulation seeds. |
| [`benchmarks/autoresearch_negative/`](https://github.com/kevinkorfmann/gamma_smc_cu/tree/main/benchmarks/autoresearch_negative) | Separate recurrent negative-selection model family and sensitivity checks. |

Each directory's README and CLI describe its inputs. Slurm wrappers preserve
the original host layout and require adaptation on other systems. Profile
matching is exploratory; it does not establish a selection mechanism or provide
calibrated biological parameter posteriors. Passing technical tests does not
validate an entire simulation study.

## Verification and scope

The published findings can be checked without GPU inference:

```bash
python3 findings_database/verify_release.py
```

The later analysis components have small CPU-only test suites. Run each in its
own Python process, because several use local modules named `common` or `metrics`:

```bash
python -m unittest discover -s analysis/rank_diagnostics -p 'test_*.py'
python -m unittest discover -s benchmarks/detection_localisation -p 'test_*.py'
python -m unittest discover -s benchmarks/autoresearch_profiles -p 'test_*.py'
python -m unittest discover -s benchmarks/autoresearch_negative -p 'test_*.py'
python -m unittest discover -s benchmarks/selection_robustness -p 'test_*.py'
```

These require the corresponding scientific Python dependencies. They do not run
the expensive empirical analysis or establish CUDA parity. The recovery package
was checksum-verified and reconstructed from a fresh GitHub clone; its full
Linux/CUDA environment builds and multi-day scientific analysis were not rerun
during packaging. Preserve new validation records alongside any rerun outputs.
