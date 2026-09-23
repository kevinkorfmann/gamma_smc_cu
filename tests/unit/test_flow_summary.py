"""Numerical and allocation regressions for fused flow summaries."""
import ctypes
import gc
import os

import numpy as np
import pytest

import gamma_smc_cu as g
from gamma_smc_cu import _core


def make_data(sites):
    rng = np.random.default_rng(819)
    G = rng.integers(0, 2, (48, sites), dtype=np.uint8)
    positions = np.cumsum(rng.integers(1, 200, sites)).astype(float)
    return G, positions


def make_pairs(count):
    # Duplicate and self pairs are valid and exercise extrema across varied values.
    pairs = [(i, j) for i in range(48) for j in range(i + 1)]
    return [pairs[i % len(pairs)] for i in range(count)]


def assert_summary(summary, means):
    assert summary["n_pairs"] == means.shape[1]
    np.testing.assert_allclose(summary["site_mean"], means.mean(axis=1, dtype=np.float64), rtol=1e-5)
    np.testing.assert_allclose(summary["site_min"], means.min(axis=1), rtol=2e-6)
    np.testing.assert_allclose(summary["site_max"], means.max(axis=1), rtol=2e-6)
    for key in ["site_mean", "site_min", "site_max"]:
        assert summary[key].dtype == np.float32
        assert np.isfinite(summary[key]).all()


@pytest.fixture(scope="module")
def ctx():
    return g.FlowContext(*make_data(65), cache_steps=64)


@pytest.mark.parametrize("count", [1, 31, 32, 33, 257, 1023, 1024, 1025, 8193, 16385, 32769, 65537])
def test_summary_matches_full_outputs(ctx, count):
    pairs = make_pairs(count)
    means = ctx.run_fb(pairs)["mean"]
    assert_summary(ctx.run_fb_summary(pairs), means)


@pytest.mark.parametrize("sites", [1, 257, 1025])
def test_summary_reused_after_other_methods(sites):
    ctx = g.FlowContext(*make_data(sites), cache_steps=64)
    pairs = make_pairs(1025)
    means = ctx.run_fb(pairs, mean_only=False, return_posterior=True)["mean"]
    assert_summary(ctx.run_fb_summary(pairs), means)
    ctx.run_fwd(make_pairs(3), mean_only=False)
    ctx.run_fb_blockwise(make_pairs(33), 61, 256, 17, 2)
    assert_summary(ctx.run_fb_summary(pairs), means)
    # Returning to full outputs must reallocate the buffers discarded by summary.
    np.testing.assert_array_equal(ctx.run_fb(pairs)["mean"], means)


@pytest.mark.parametrize("sites,count", [(65, 0), (0, 0), (0, 33)])
def test_summary_empty_inputs(sites, count):
    ctx = g.FlowContext(*make_data(sites), cache_steps=64)
    summary = ctx.run_fb_summary(make_pairs(count))
    assert summary["n_pairs"] == count
    for key in ["site_mean", "site_min", "site_max"]:
        assert summary[key].shape == (sites,)
        assert np.isnan(summary[key]).all()


def test_summary_does_not_retain_dense_output():
    sites, count = 2048, 1024
    ctx = g.FlowContext(*make_data(sites), cache_steps=64)
    ctx.run_fb_summary(make_pairs(1))  # warm CUDA module loading before measuring
    before = _core.cuda_mem_info()[0]
    ctx.run_fb_summary(make_pairs(count))
    retained = before - _core.cuda_mem_info()[0]
    # Two forward-state planes, pair indices, and allocator rounding; no third plane.
    assert retained <= 2 * sites * count * 4 + 2 * 1024**2


@pytest.mark.skipif(os.environ.get("GAMMA_SMC_CU_LARGE_TESTS") != "1",
                    reason="opt-in GPU memory pressure test")
def test_summary_multiple_chunks_under_memory_pressure():
    data = make_data(1025)
    pairs = make_pairs(4001)
    reference = g.FlowContext(*data, cache_steps=64)
    means = reference.run_fb(pairs)["mean"]
    ctx = g.FlowContext(*data, cache_steps=64)
    del reference
    gc.collect()
    cuda = ctypes.CDLL("libcudart.so")
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    free, _ = _core.cuda_mem_info()
    if free < 1024**3:
        pytest.skip("requires at least 1 GiB free on an idle GPU")
    reservation = ctypes.c_void_p()
    assert cuda.cudaMalloc(ctypes.byref(reservation), free - 524 * 1024**2) == 0
    try:
        # Only ~12 MiB above the planner's reserve: the 32 MiB forward states
        # must be split, including a partial last chunk and both reduction paths.
        for _ in range(2):
            assert_summary(ctx.run_fb_summary(pairs), means)
    finally:
        assert cuda.cudaFree(reservation) == 0
