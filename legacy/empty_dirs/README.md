# Archived empty directories

These directories existed in the working tree but contained no files.
They were placeholders from earlier scaffolding and are not used by any
current v4.1 script, figure, table, verify, or build target. They are
preserved here (with `.gitkeep`) only for historical reference.

- `logs/` — top-level log dir; unused (per-analysis `logs/` still live
  under `analysis/*/logs/` and are gitignored)
- `tests/performance/` — never populated
- `tests/regression/golden/` — never populated (parent `tests/regression/`
  was also empty and removed)
- `src/io/` — never populated; not referenced by `CMakeLists.txt`
- `analysis/relate_clues/figures/` — outputs live elsewhere
- `benchmarks/pairwise_scaling/gsmc_verify/` — never populated

Separately: the empty top-level `docs_local/` was removed outright and
added to `.gitignore`. It is still referenced as an output path by
`private/manuscript/v4.1/figures/gen_fig_*.py` — those scripts write
into it locally; the directory is recreated on demand.
