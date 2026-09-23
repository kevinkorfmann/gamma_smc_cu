"""Regression tests for cache ownership, CUDA bounds, and posterior summaries.

The large-allocation checks are opt-in: GAMMA_SMC_CU_LARGE_TESTS=1. Run them
only on an otherwise idle GPU with at least 8 GiB of free memory.
"""
import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import gamma_smc_cu as g
from gamma_smc_cu import _core
from gamma_smc_cu.infer import _resolve_flow_field_path


@pytest.fixture
def data():
    rng = np.random.default_rng(4281)
    G = rng.integers(0, 2, (16, 257), dtype=np.uint8)
    G[0] = 0
    G[1] = 1
    return G, np.arange(G.shape[1], dtype=float) * 10 + 100_000


def context(data, **kwargs):
    return g.FlowContext(*data, cache_steps=64, **kwargs)


def test_live_context_survives_different_parameters(data):
    a = context(data)
    expected = a.run_fb([(2, 3)])["mean"].copy()
    b = context(data, mu=5e-8)
    changed = b.run_fb([(2, 3)])["mean"]
    assert not np.allclose(expected, changed)
    np.testing.assert_array_equal(a.run_fb([(2, 3)])["mean"], expected)
    del b
    gc.collect()
    np.testing.assert_array_equal(a.run_fb([(2, 3)])["mean"], expected)


def test_flow_field_path_and_content_are_honored(data, tmp_path):
    field = tmp_path / "field.txt"
    field.write_text(Path(_resolve_flow_field_path(None)).read_text())
    a = context(data, flow_field_path=str(field))
    expected = a.run_fb([(2, 3)])["mean"].copy()
    with pytest.raises(ValueError, match="load flow field"):
        context(data, flow_field_path=str(tmp_path / "missing.txt"))
    lines = field.read_text().splitlines()
    field.write_text("\n".join(lines[:2]) + "\n" + "\n".join(
        " ".join(str(float(value) * 2) for value in line.split()) for line in lines[2:]) + "\n")
    b = context(data, flow_field_path=str(field))
    assert not np.allclose(expected, b.run_fb([(2, 3)])["mean"])
    np.testing.assert_array_equal(a.run_fb([(2, 3)])["mean"], expected)
    field.write_text("invalid field")
    with pytest.raises(ValueError, match="load flow field"):
        context(data, flow_field_path=str(field))


@pytest.mark.parametrize("spacing,start", [(1., 0.), (1., 100_000.), (512., 32.)])
def test_forward_matches_forward_states(data, spacing, start):
    G, positions = data
    positions = np.arange(len(positions), dtype=float) * spacing + start
    ctx = context((G, positions))
    pairs = [(0, 1), (2, 3), (4, 4)]
    states = ctx.run_fwd_states(pairs)
    result = ctx.run_fwd(pairs, mean_only=False)
    mean = 10. ** states["mean_log10"].astype(float) * 20_000
    alpha = 10. ** (-2. * states["cv_log10"].astype(float))
    base, spread = 1 - 1 / (9 * alpha), 1.96 / np.sqrt(9 * alpha)
    np.testing.assert_allclose(result["mean"], mean, rtol=2e-6)
    np.testing.assert_allclose(result["lower"], mean * np.maximum(0, base - spread)**3, rtol=2e-5)
    np.testing.assert_allclose(result["upper"], mean * (base + spread)**3, rtol=2e-5)
    direct = g.gamma_smc_flow_cached_fwd(G, positions, pairs, cache_steps=64, mean_only=False)
    for key in result:
        np.testing.assert_array_equal(direct[key], result[key])


@pytest.mark.parametrize("ci,posterior", [(False, False), (True, False), (True, True)])
def test_blockwise_outputs_and_streams(data, ci, posterior):
    ctx = context(data)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    expected = ctx.run_fb(pairs, mean_only=not ci, return_posterior=posterior)
    # Every padded block covers the sequence, so this is an exact reference.
    for streams in ([1] if posterior else [1, 2, 4]):
        result = ctx.run_fb_blockwise(pairs, core_block_sites=61, flank_sites=1000,
                                     pair_batch_size=2, max_streams=streams,
                                     mean_only=not ci, return_posterior=posterior)
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])


def test_context_calls_are_serialized_safely(data):
    ctx = context(data)
    pairs = [[(0, 1)], [(2, 3), (4, 5)]]
    expected = [ctx.run_fb(p)["mean"].copy() for p in pairs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ctx.run_fb, p) for p in pairs]
        for want, future in zip(expected, futures):
            np.testing.assert_array_equal(future.result()["mean"], want)


@pytest.mark.parametrize("pair", [(-1, 0), (0, 16), (99999, 0)])
def test_native_pair_bounds(data, pair):
    ctx = context(data)
    for method in [ctx.run_fwd, ctx.run_fwd_states, ctx.run_fb, ctx.run_fb_summary, ctx.run_fb_blockwise]:
        with pytest.raises(ValueError, match="Pair indices"):
            method([pair])
    for method in [g.pairwise_prefix_scan, g.gamma_smc_flow_cached_fb, g.gamma_smc_flow_cached_fwd]:
        args = (data[0], [pair]) if method == g.pairwise_prefix_scan else (*data, [pair])
        with pytest.raises(ValueError, match="Pair indices"):
            method(*args)
    with pytest.raises(ValueError, match="Pair indices"):
        g.hmm_posterior(*data, pair)
    hmm = g.HMMContext(*data)
    with pytest.raises(ValueError, match="Pair indices"):
        hmm.run_batch([pair])


@pytest.mark.parametrize("allele", [-1, 2, 256, .5, np.nan])
def test_reject_unsupported_alleles_before_uint8_conversion(data, allele):
    G, positions = data
    G = G.astype(float)
    G[3, 0] = allele
    for infer in [g.infer, g.infer_blockwise]:
        with pytest.raises(ValueError, match="0 and 1"):
            infer(G, positions, pairs=[(0, 1)])


@pytest.mark.parametrize("pair", [(0.5, 1), (-1, 0), (16, 0)])
def test_high_level_pair_validation(data, pair):
    for infer in [g.infer, g.infer_blockwise]:
        with pytest.raises(ValueError):
            infer(*data, pairs=[pair])


def test_native_shape_and_value_validation(data):
    G, positions = data
    with pytest.raises(ValueError):
        context((G.ravel(), positions))
    with pytest.raises(ValueError):
        context((G, positions[:-1]))
    for value in [-1., np.nan, np.inf, positions[1]]:
        bad = positions.copy()
        bad[0] = value
        with pytest.raises(ValueError):
            context((G, bad))
    for kwargs in [dict(Ne=0), dict(mu=-1), dict(rho=np.nan)]:
        with pytest.raises(ValueError):
            context(data, **kwargs)
    bad = G.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError):
        g.bitpack(bad)
    packed = g.bitpack(G)
    for n, sites in [(17, len(positions)), (16, 1000), (16, -1)]:
        with pytest.raises(ValueError):
            g.unpack(packed, n, sites)


@pytest.mark.parametrize("blockwise", [False, True])
def test_monomorphic_input_returns_requested_empty_arrays(blockwise):
    infer = g.infer_blockwise if blockwise else g.infer
    kwargs = dict(verbose=True) if blockwise else {}
    result = infer(np.zeros((4, 8), dtype=np.uint8), np.arange(8, dtype=float),
                   pairs=[(0, 1)], mean_only=False, return_posterior=True, **kwargs)
    for key in ["mean", "lower", "upper", "posterior_alpha", "posterior_beta"]:
        assert result[key].shape == (0, 1)
    if blockwise:
        assert result["blocks"].shape == (0, 4)


def test_native_empty_arrays(data):
    for dataset in [data, (data[0][:, :0].copy(), data[1][:0])]:
        ctx = context(dataset)
        for pairs in [[], [(0, 1)]]:
            if pairs and len(dataset[1]):
                continue
            for method in [ctx.run_fb, ctx.run_fb_blockwise]:
                r = method(pairs, mean_only=False, return_posterior=True)
                for key in ["mean", "lower", "upper", "posterior_alpha", "posterior_beta"]:
                    assert r[key].shape == (len(dataset[1]), len(pairs))
    assert g.bitpack(data[0][:, :0].copy()).shape == (16, 0)


@pytest.mark.parametrize("K", [32, 64, 128])
def test_hmm_last_site_quantiles_and_context(K):
    G = np.zeros((4, 8), dtype=np.uint8)
    G[3] = 1
    positions = np.arange(8, dtype=float)
    gamma = g.hmm_posterior(G, positions, (0, 1), K=K)
    times = g.time_midpoints(K=K)
    cdf = np.cumsum(gamma, axis=1)
    want_lower = times[(cdf >= .025).argmax(axis=1)]
    want_upper = times[(cdf >= .975).argmax(axis=1)]
    _, mean, lower, upper, _ = g.hmm_posterior_batched(G, positions, [(0, 1)], K=K)
    np.testing.assert_allclose(mean[0], gamma @ times, rtol=2e-6)
    np.testing.assert_allclose(lower[0], want_lower, rtol=2e-6)
    np.testing.assert_allclose(upper[0], want_upper, rtol=2e-6)
    ctx = g.HMMContext(G, positions, K=K)
    for count in [1, 3, 1]:
        result = ctx.run_batch([(0, 1)] * count)
        np.testing.assert_allclose(result[0], np.repeat(mean, count, axis=0), rtol=2e-6)


def test_ep_mean_includes_upper_tail():
    G = np.zeros((4, 8), dtype=np.uint8)
    result = _core.ep_infer(G, np.arange(8, dtype=float), [(1, 0), (2, 0), (2, 1)],
                            m_haplotypes=3, K=32, max_iterations=1)
    np.testing.assert_allclose(result["mean"], result["gamma"] @ g.time_midpoints(K=32), rtol=2e-6)


@pytest.mark.skipif(_core.get_device_count() < 2, reason="requires two GPUs")
def test_secondary_device_cache_memory_stabilizes(data):
    try:
        for device in [0, 1]:
            _core.set_device(device)
            for _ in range(2):
                ctx = context(data)
                ctx.run_fb([(0, 1)])
                del ctx
            gc.collect()
            before = _core.cuda_mem_info()[0]
            for _ in range(5):
                ctx = context(data)
                ctx.run_fb([(0, 1)])
                del ctx
            gc.collect()
            after = _core.cuda_mem_info()[0]
            assert before - after < 1024**2
    finally:
        _core.set_device(0)


large = pytest.mark.skipif(os.environ.get("GAMMA_SMC_CU_LARGE_TESTS") != "1",
                          reason="opt-in GPU/host memory stress test")


@large
def test_bitpack_unpack_above_signed_32_bit_offset():
    G = np.zeros((4097, 524288), dtype=np.uint8)
    G[-1, 0] = 1
    G[-1, -1] = 1
    packed = g.bitpack(G)
    assert packed[-1, 0] == 1
    assert packed[-1, -1] == (1 << 63)
    unpacked = g.unpack(packed, *G.shape)
    np.testing.assert_array_equal(unpacked[-1], G[-1])
    assert np.count_nonzero(unpacked) == 2


@large
def test_multistream_memory_bounded_by_block():
    rng = np.random.default_rng(123)
    G = rng.integers(0, 2, (128, 32768), dtype=np.uint8)
    positions = np.arange(G.shape[1], dtype=float) * 10
    pairs = [(i, j) for i in range(1, 128) for j in range(i)][:7168]
    ctx = context((G, positions))
    cuda = ctypes.CDLL("libcudart.so")
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    free, _ = _core.cuda_mem_info()
    if free < 8 * 1024**3:
        pytest.skip("requires at least 8 GiB free on an idle GPU")
    reservation = ctypes.c_void_p()
    assert cuda.cudaMalloc(ctypes.byref(reservation), free - 768 * 1024**2) == 0
    try:
        a = ctx.run_fb_blockwise(pairs, 2048, 256, 256, 1)["mean"]
        b = ctx.run_fb_blockwise(pairs, 2048, 256, 256, 2)["mean"]
        # The host result alone exceeds the remaining free GPU memory.
        assert a.nbytes > _core.cuda_mem_info()[0]
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(b).all()
    finally:
        cuda.cudaFree(reservation)


@pytest.mark.skipif(_core.get_device_count() < 2, reason="requires two GPUs")
def test_hmm_context_remembers_its_device(data):
    try:
        _core.set_device(0)
        ctx = g.HMMContext(*data)
        expected = ctx.run_batch([(0, 1)])[0].copy()
        _core.set_device(1)
        np.testing.assert_array_equal(ctx.run_batch([(0, 1)])[0], expected)
        assert _core.get_device() == 1
        del ctx
        assert _core.get_device() == 1
    finally:
        _core.set_device(0)
