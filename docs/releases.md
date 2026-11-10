# Maintaining releases

## 2026-11-10 · Analysis source and recovery documentation

Added previously local rank-diagnostic, detection/localisation, exploratory
profile-matching and selection/BGS follow-up scripts. The
[reproduction guide](reproducing.md) now identifies the corrected full-chromosome
protocol, current collector, final CLUES correction chain and the separate
private recovery repository. Its source archives and run-directory reconstruction
were verified from a fresh GitHub clone. Full scientific reruns and fresh CUDA
environment builds were not performed as part of the recovery packaging.

The September findings tables and their immutable release hashes are unchanged.

## 2026-09-27

The first release on this rebuilt documentation site publishes the corrected
143-locus atlas, calibrated accuracy and runtime tables, threshold sensitivity,
and focal CLUES2 summaries. It labels the earlier 165-locus deposit as historical,
and adds the executed iHS/nSL methods.
The findings page also includes regional iHS plots for GRK2 and TREML1 / TREM2,
with downloadable plotted scores and source hashes. These documentation assets
supplement the release; the published atlas tables are unchanged.
See [Latest findings](findings.md).

## Add a findings release

1. Create a new dated directory under `findings_database/releases/`. Preserve
   previous release bytes; corrections receive a new directory and a clear notice.
2. Include the tables, a data dictionary, source identity, file hashes, and the
   scope of verification. Describe missing raw inputs and unresolved validation.
3. Run the release verification script and update `LATEST.json`, the findings
   page and this release history together.
4. Build the docs with warnings treated as errors. Publish the checked commit
   and a release tag; record that tag in any manuscript citing the deposit.

Do not reuse a historical table simply because its filename matches a new one.
Do not put personal paths, credentials or private working notes in a release.

## Build the site locally

```bash
python -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python findings_database/verify_release.py
sphinx-build -n -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

The docs build is CPU-only and uses pinned direct dependencies. The repository's
GitHub Actions workflow builds the same source and checks the release manifest.

## Read the Docs

Import the public repository
`https://github.com/kevinkorfmann/gamma_smc_cu` into Read the Docs Community,
use `main` as the default branch and `.readthedocs.yaml` as the configuration.
If the `gamma-smc-cu` project already exists in the account, use that project.
Enable its GitHub integration so pushes rebuild `latest`; activate release tags
when an archived documentation version is needed.

The checked-in configuration specifies the OS, Python version, requirements
and strict Sphinx build according to the
[Read the Docs configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
Successful local or GitHub builds do not by themselves prove that the hosted
Read the Docs project is connected or has published the same commit.
