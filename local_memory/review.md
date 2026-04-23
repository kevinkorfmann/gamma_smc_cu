# Review Findings

The new top-level `infer()` wrapper is not safe to ship as-is: its advertised default path cannot locate a usable flow-field file on normal installs, its all-pairs default drives the unchunked GPU path into OOM on common workloads, and malformed matrix inputs can crash inside the binding layer instead of failing fast.

These issues also look fairly straightforward to address. They are primarily packaging, default-behavior, and input-validation problems in the new wrapper layer, not direct evidence that the core inference implementation is numerically wrong for valid, explicitly configured inputs. This review does not establish full end-to-end correctness; it only narrows the concern to wrapper safety and release readiness.

## Findings

- [P1] Bundle a default flow field or require `flow_field_path`
  File: `python/gamma_smc_cu/infer.py:59-61`

  The new `gamma_smc_cu.infer(ts)` path is advertised as a one-line API, but this fallback only checks a package-local `default_flow_field.txt` that is not present in this repo and a hard-coded `/sietch_colab/...` path. On any normal install, `flow_field_path=None` therefore raises `FileNotFoundError`, so the new default example never works unless the caller already has the author's private file layout.

- [P1] Avoid defaulting `infer()` to an unchunked all-pairs workload
  File: `python/gamma_smc_cu/infer.py:52-53`

  When `pairs` is omitted, this expands to every haplotype pair and then sends the whole list into `FlowContext.run_fb()`, whose C++ binding still allocates `O(S * n_pairs)` buffers in one shot. On the README-style `infer(ts)` call, a fairly ordinary input like 200 haplotypes x 100k sites already needs about 24 GB just for `d_fwd_buf` and `d_mean`, so the new default OOMs on 24 GB cards before caches or outputs are counted.

- [P2] Validate matrix inputs before passing them to `_core.FlowContext`
  File: `python/gamma_smc_cu/infer.py:46-47`

  For matrix inputs we do not validate `positions` at all before calling into the pybind layer. `np.ascontiguousarray(None, dtype=np.float64)` becomes a length-1 array, and any length-mismatched `positions` has the same effect: `FlowContext` then blindly copies `S` doubles from that buffer, which can read past the NumPy allocation and crash the interpreter instead of raising a `ValueError`.
