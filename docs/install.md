# Installation

The supported build environment is **Linux x86-64 with an NVIDIA GPU**.
The Python package contains a compiled CUDA extension; installing the docs
or copying the Python directory alone does not install that extension.

## Build from source

Install [Pixi](https://pixi.sh/latest/installation/) and an NVIDIA driver
compatible with the CUDA toolkit, then clone the repository:

```bash
git clone https://github.com/kevinkorfmann/gamma_smc_cu.git
cd gamma_smc_cu
pixi install
pixi shell
```

Inside that shell, configure for your GPU, build, and install the two shared
libraries into the Python package:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build build --parallel 4
cmake --install build
python -c "from gamma_smc_cu import infer, export_dense; print('Import OK')"
```

`80` targets A100. Use `70` for the V100/V100S used in the reported benchmark;
choose the architecture appropriate to other hardware. The short `pixi run build`
task currently targets `80`, so use the explicit commands above for a V100.

The checked-in Pixi environment constrains the toolkit to CUDA 12.0–12.2.
It does not produce native Blackwell cubins. For a native B200 build, use
CUDA 12.8 or later and architecture `100`. Earlier-toolkit builds can instead
run through compatible PTX; that is a different compilation path. See the
[NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).
Record the toolkit and architecture with benchmarks, and do not use the V100
runtime result as a B200 performance estimate.

The Pixi shell sets `PYTHONPATH` to this checkout's `python/` directory and
adds its shared-library directory to `LD_LIBRARY_PATH`. Keep that shell active
when running the examples. On a cluster, run builds and GPU work in an allocation
under the site's scheduler.

## Check the installation

Run the [small example](python-api.md#a-small-inference), then the relevant GPU tests:

```bash
python -m pytest tests/unit/ -q
```

GPU tests need a visible compatible device. Documentation builds need neither
CUDA nor the compiled package; see [Maintaining releases](releases.md).

## Common problems

| Symptom | Check |
| --- | --- |
| Cannot import `_core` | Run `cmake --install build` and use the Pixi shell. |
| Cannot find `libgamma_smc_cu_kernels.so` | Check the installed library beside `_core` and `LD_LIBRARY_PATH`. |
| No kernel image for this device | Reconfigure and rebuild for the correct CUDA architecture. |
| Out of GPU memory | Reduce `pair_batch_size`; use checkpointed batches or interval summaries. |
| Out of host memory or disk | Dense means alone require approximately `4 × sites × pairs` bytes; stream or summarize. |
