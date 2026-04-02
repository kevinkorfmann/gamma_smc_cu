"""
Integration tests for MultiGPUFlowContext.

Validates that multi-GPU results match single-GPU results for:
  - run_fwd (forward-only)
  - run_fb (forward-backward)
  - run_fb_summary (per-site reduction)

Skips automatically if fewer than 2 GPUs are available.
"""

import numpy as np
import pytest

import tmrca_cu._core as _core
from tmrca_cu.multigpu import MultiGPUFlowContext

FLOW_FIELD_PATH = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"

N_GPUS = _core.get_device_count()

pytestmark = pytest.mark.skipif(N_GPUS < 2, reason=f"need >= 2 GPUs, have {N_GPUS}")


@pytest.fixture(scope="module")
def sim_data():
    """Simulate a small dataset for testing."""
    rng = np.random.default_rng(42)
    n, S = 20, 5000
    G = rng.integers(0, 2, size=(n, S), dtype=np.uint8)
    positions = np.sort(rng.uniform(0, 1_000_000, size=S))
    pairs = [(i, j) for i in range(1, 10) for j in range(i)]
    return G, positions, pairs


@pytest.fixture(scope="module")
def single_gpu_ctx(sim_data):
    """FlowContext on GPU 0 only."""
    G, positions, _ = sim_data
    _core.set_device(0)
    return _core.FlowContext(G, positions, 10000.0, 1.25e-8, 1e-8,
                              FLOW_FIELD_PATH, 0)


@pytest.fixture(scope="module")
def multi_gpu_ctx(sim_data):
    """MultiGPUFlowContext across all GPUs."""
    G, positions, _ = sim_data
    return MultiGPUFlowContext(G, positions, 10000.0, 1.25e-8, 1e-8,
                                FLOW_FIELD_PATH)


def test_run_fwd_matches(sim_data, single_gpu_ctx, multi_gpu_ctx):
    """Multi-GPU run_fwd should produce same results as single-GPU."""
    _, _, pairs = sim_data

    single = single_gpu_ctx.run_fwd(pairs, mean_only=True)
    multi = multi_gpu_ctx.run_fwd(pairs, mean_only=True)

    np.testing.assert_allclose(
        multi["mean"], single["mean"], rtol=1e-5, atol=1e-6,
        err_msg="run_fwd mean mismatch between single and multi GPU")


def test_run_fb_matches(sim_data, single_gpu_ctx, multi_gpu_ctx):
    """Multi-GPU run_fb should produce same results as single-GPU."""
    _, _, pairs = sim_data

    single = single_gpu_ctx.run_fb(pairs, mean_only=True)
    multi = multi_gpu_ctx.run_fb(pairs, mean_only=True)

    np.testing.assert_allclose(
        multi["mean"], single["mean"], rtol=1e-5, atol=1e-6,
        err_msg="run_fb mean mismatch between single and multi GPU")


def test_run_fb_with_ci(sim_data, single_gpu_ctx, multi_gpu_ctx):
    """Multi-GPU run_fb with CI should match single-GPU."""
    _, _, pairs = sim_data

    single = single_gpu_ctx.run_fb(pairs, mean_only=False)
    multi = multi_gpu_ctx.run_fb(pairs, mean_only=False)

    for key in ("mean", "lower", "upper"):
        np.testing.assert_allclose(
            multi[key], single[key], rtol=1e-5, atol=1e-6,
            err_msg=f"run_fb {key} mismatch between single and multi GPU")


def test_run_fb_summary_matches(sim_data, single_gpu_ctx, multi_gpu_ctx):
    """Multi-GPU run_fb_summary should produce same site means."""
    _, _, pairs = sim_data

    single = single_gpu_ctx.run_fb_summary(pairs)
    multi = multi_gpu_ctx.run_fb_summary(pairs)

    np.testing.assert_allclose(
        multi["site_mean"], single["site_mean"], rtol=1e-4, atol=1e-5,
        err_msg="run_fb_summary site_mean mismatch")


def test_uses_all_gpus(multi_gpu_ctx):
    """Verify contexts were created on different devices."""
    device_ids = set()
    for ctx in multi_gpu_ctx.contexts:
        device_ids.add(ctx.device_id)
    assert len(device_ids) == multi_gpu_ctx.n_gpus


def test_empty_pairs(multi_gpu_ctx):
    """Empty pair list should return empty arrays without error."""
    result = multi_gpu_ctx.run_fwd([], mean_only=True)
    assert result["mean"].shape[1] == 0

    result = multi_gpu_ctx.run_fb([], mean_only=True)
    assert result["mean"].shape[1] == 0

    result = multi_gpu_ctx.run_fb_summary([])
    assert result["n_pairs"] == 0


def test_run_fwd_with_ci(sim_data, single_gpu_ctx, multi_gpu_ctx):
    """Multi-GPU run_fwd with CI should match single-GPU."""
    _, _, pairs = sim_data

    single = single_gpu_ctx.run_fwd(pairs, mean_only=False)
    multi = multi_gpu_ctx.run_fwd(pairs, mean_only=False)

    for key in ("mean", "lower", "upper"):
        np.testing.assert_allclose(
            multi[key], single[key], rtol=1e-5, atol=1e-6,
            err_msg=f"run_fwd {key} mismatch")
