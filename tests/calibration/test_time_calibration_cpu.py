"""Calibration contract without CUDA or an installed extension.

Run standalone with ``pytest --confcutdir=tests/calibration tests/calibration``.
The mock supplies a known dimensionless posterior and the native 2Ne output
contract, so these tests exercise the complete public wrapper conversion.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def api(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "python/gamma_smc_cu/infer.py"
    spec = importlib.util.spec_from_file_location("calibration_infer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contexts = []

    class MockFlowContext:
        def __init__(self, G, positions, Ne, mu, rho, *args):
            self.n_sites, self.Ne = len(positions), Ne
            self.scaled_mu, self.scaled_rho = 4 * Ne * mu, 4 * Ne * rho
            contexts.append(self)

        def run_fb(self, pairs, mean_only=True, return_posterior=False):
            shape = (self.n_sites, len(pairs))
            result = {"mean": np.full(shape, 0.75 * 2 * self.Ne, np.float32)}
            if not mean_only:
                result.update(
                    lower=np.full(shape, 0.25 * 2 * self.Ne, np.float32),
                    upper=np.full(shape, 1.5 * 2 * self.Ne, np.float32))
            if return_posterior:
                result.update(posterior_alpha=np.full(shape, 3.0, np.float32),
                              posterior_beta=np.full(shape, 4.0, np.float32))
            return result

        def run_fb_blockwise(self, pairs, mean_only=True, return_posterior=False, **kw):
            result = self.run_fb(pairs, mean_only, return_posterior)
            result["blocks"] = np.array([[0, self.n_sites, 0, self.n_sites]], np.int32)
            return result

    monkeypatch.setitem(sys.modules, "gamma_smc_cu", SimpleNamespace(
        _core=SimpleNamespace(FlowContext=MockFlowContext)))
    monkeypatch.setattr(module, "_query_free_gpu_bytes", lambda: None)
    return module, contexts


@pytest.fixture(params=["infer", "infer_blockwise"])
def run(api, request):
    module, contexts = api

    def decode(G=None, positions=None, **kwargs):
        # Diploid heterozygosity counts 2 and 4 over a 10,000 bp span:
        # theta = ((2 + 4) / 2) / 10,000 = 0.0003.
        if G is None:
            G = np.array([[0, 0, 0, 0], [1, 1, 0, 0],
                          [0, 0, 0, 0], [1, 1, 1, 1]], np.uint8)
            positions = np.array([9, 999, 4999, 9999], np.float64)
        options = dict(pairs=[(0, 1)], mu=1e-8, rho=2e-8, Ne=10000,
                       mean_only=False, return_posterior=True,
                       flow_field_path="mock-flow-field")
        if request.param == "infer_blockwise":
            options.update(core_block_sites=max(1, len(positions)), flank_sites=0)
        options.update(kwargs)
        return getattr(module, request.param)(G, positions, **options)

    return decode, contexts


def test_physical_mean_bounds_and_posterior(run):
    decode, contexts = run
    result = decode()
    # theta / (2*physical_mu) = 15,000 generations per coalescent unit.
    for key, expected in [("mean", 11250), ("lower", 3750), ("upper", 22500)]:
        np.testing.assert_allclose(result[key], expected)
        assert result[key].dtype == np.float32
    np.testing.assert_array_equal(result["posterior_alpha"], 3)
    np.testing.assert_array_equal(result["posterior_beta"], 4)
    metadata = result["metadata"]
    assert metadata["scaled_mutation_rate"] == pytest.approx(0.0003)
    assert metadata["scaled_recombination_rate"] == pytest.approx(0.0006)
    assert metadata["generations_per_coalescent_unit"] == pytest.approx(15000)
    assert metadata["effective_Ne"] == pytest.approx(7500)
    assert metadata["generation_rescaling_factor"] == pytest.approx(0.75)
    assert metadata["physical_mu"] == 1e-8
    assert metadata["time_units"] == "generations"
    assert metadata["posterior_time_units"] == "coalescent"
    assert contexts[-1].scaled_mu == pytest.approx(0.0003)
    assert contexts[-1].scaled_rho == pytest.approx(0.0006)
    json.dumps(metadata, allow_nan=False)


def test_auto_calibration_is_independent_of_bookkeeping_Ne(run):
    decode, contexts = run
    first, second = decode(Ne=10000), decode(Ne=40000)
    for key in ("mean", "lower", "upper", "posterior_alpha", "posterior_beta"):
        np.testing.assert_array_equal(first[key], second[key])
    assert contexts[0].scaled_mu == contexts[1].scaled_mu
    assert contexts[0].scaled_rho == contexts[1].scaled_rho


def test_physical_mutation_rate_sets_inverse_time_scale(run):
    decode, _ = run
    first, second = decode(), decode(mu=2e-8, rho=4e-8)
    for key in ("mean", "lower", "upper"):
        np.testing.assert_allclose(first[key], 2 * second[key])
    assert first["metadata"]["scaled_mutation_rate"] == second["metadata"]["scaled_mutation_rate"]
    assert first["metadata"]["scaled_recombination_rate"] == second["metadata"]["scaled_recombination_rate"]


def test_fixed_parameters_keep_existing_2Ne_output(run):
    decode, _ = run
    result = decode(auto_estimate_theta=False)
    np.testing.assert_array_equal(result["mean"], 15000)
    assert result["metadata"]["generations_per_coalescent_unit"] == 20000
    assert result["metadata"]["time_scale_source"] == "fixed_Ne"
    assert result["metadata"]["physical_mu"] is None


def test_manual_effective_rates_can_use_physical_calibration(run):
    decode, _ = run
    auto = decode()
    manual = decode(mu=7.5e-9, rho=1.5e-8, auto_estimate_theta=False,
                    physical_mu=1e-8)
    for key in ("mean", "lower", "upper", "posterior_alpha", "posterior_beta"):
        np.testing.assert_array_equal(auto[key], manual[key])
    assert manual["metadata"]["time_scale_source"] == "physical_mutation_rate"


def test_empty_result_records_scale_without_native_context(run):
    decode, contexts = run
    result = decode(G=np.zeros((4, 0), np.uint8), positions=np.array([]))
    assert result["mean"].shape == (0, 1)
    assert result["metadata"]["generations_per_coalescent_unit"] == 20000
    assert not contexts


@pytest.mark.parametrize("physical_mu", [0, -1, float("nan"), float("inf")])
def test_invalid_physical_rate_rejected_before_native_context(run, physical_mu):
    decode, contexts = run
    with pytest.raises(ValueError, match="physical_mu"):
        decode(physical_mu=physical_mu)
    assert not contexts


def test_zero_mu_requires_fixed_parameter_mode(run):
    decode, _ = run
    with pytest.raises(ValueError, match="mu must be positive"):
        decode(mu=0)
    result = decode(mu=0, auto_estimate_theta=False)
    assert result["metadata"]["generations_per_coalescent_unit"] == 20000
