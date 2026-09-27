"""The streaming path must preserve full-sequence calibrated inference."""
from contextlib import closing

import numpy as np
import pytest
import gamma_smc_cu as g
from gamma_smc_cu.batching import iter_infer_batches


@pytest.fixture
def panel():
    rng = np.random.default_rng(8725521)
    G = rng.integers(0, 2, (12, 257), dtype=np.uint8)
    positions = np.cumsum(rng.integers(1, 100, 257)).astype(float)
    pairs = [(0, 1), (5, 9), (2, 7), (1, 11), (4, 8)]
    return G, positions, pairs


@pytest.mark.parametrize("devices", [[0], [0, 1]])
@pytest.mark.parametrize("posterior", [False, True])
def test_streaming_matches_calibrated_full_context(panel, devices, posterior):
    if g._core.get_device_count() < len(devices):
        pytest.skip("requires two GPUs")
    G, positions, pairs = panel
    kwargs = dict(mu=2.1e-8, rho=0.9e-8, physical_mu=1.25e-8,
                  auto_estimate_theta=False, mean_only=False,
                  return_posterior=posterior)
    reference = g.infer_blockwise(G, positions, pairs=pairs,
                                 core_block_sites=len(positions), flank_sites=0,
                                 pair_batch_size=2, **kwargs)
    batches = list(iter_infer_batches(G, positions, pairs=pairs,
                                     pair_batch_size=2, gpu_ids=devices, **kwargs))
    assert [b["pair_offset"] for b in batches] == [0, 2, 4]
    assert sum((b["pairs"] for b in batches), []) == pairs
    for b in batches:
        np.testing.assert_array_equal(b["positions"], reference["positions"])
        assert b["metadata"] == reference["metadata"]
    for key in ["mean", "lower", "upper"] + (
            ["posterior_alpha", "posterior_beta"] if posterior else []):
        np.testing.assert_array_equal(np.concatenate([b[key] for b in batches], axis=1),
                                      reference[key])


def test_early_close_and_arrays_remain_owned(panel):
    G, positions, pairs = panel
    with closing(iter_infer_batches(G, positions, pairs=pairs, pair_batch_size=1)) as stream:
        first = next(stream)
        saved = first["mean"].copy()
        next(stream)
        np.testing.assert_array_equal(first["mean"], saved)
    np.testing.assert_array_equal(first["mean"], saved)


def test_empty_and_monomorphic(panel):
    G, positions, pairs = panel
    assert list(iter_infer_batches(G, positions, pairs=[])) == []
    batches = list(iter_infer_batches(np.zeros_like(G), positions, pairs=pairs,
                                     pair_batch_size=2, auto_estimate_theta=False))
    assert [b["mean"].shape for b in batches] == [(0, 2), (0, 2), (0, 1)]


@pytest.mark.parametrize('checkpoint_sites', [31, 256])
def test_streaming_checkpoint_calibration(panel, checkpoint_sites):
    G, positions, pairs = panel
    kwargs = dict(mu=2.1e-8, rho=0.9e-8, physical_mu=1.25e-8,
                  auto_estimate_theta=False, mean_only=False, return_posterior=True)
    expected = g.infer_blockwise(G, positions, pairs=pairs,
        core_block_sites=len(positions), flank_sites=0, **kwargs)
    batches = list(g.iter_infer_batches(G, positions, pairs=pairs,
        pair_batch_size=2, checkpoint_sites=checkpoint_sites, **kwargs))
    for key in ('mean', 'lower', 'upper', 'posterior_alpha', 'posterior_beta'):
        np.testing.assert_array_equal(np.concatenate([b[key] for b in batches], axis=1),
                                      expected[key])


def test_public_region_context_filters_and_calibrates(panel):
    G, positions, pairs = panel
    G[:, [0, 31, 256]] = 0
    kwargs = dict(mu=2.1e-8, rho=0.9e-8, physical_mu=1.25e-8,
                  auto_estimate_theta=False)
    expected = g.infer_blockwise(G, positions, pairs=pairs,
        core_block_sites=len(positions), flank_sites=0, **kwargs)
    context = g.RegionMomentContext(G, positions, **kwargs)
    np.testing.assert_array_equal(context.positions, expected['positions'])
    regions = [(0, 2), (3, 65), (0, len(context.positions))]
    actual = context.run(pairs, regions, tile_sites=31)
    for i, (start, stop) in enumerate(regions):
        values = expected['mean'][start:stop].astype(np.float64)
        np.testing.assert_allclose(actual['region_mean'][i], values.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(actual['region_mean_log'][i], np.log(values).mean(axis=0), rtol=1e-12)
    assert actual['metadata'] == expected['metadata']
    assert actual['pairs'] == pairs
    assert context.run(pairs, [])['region_mean'].shape == (0, len(pairs))
    assert context.run([], regions)['region_mean_log'].shape == (len(regions), 0)
    with pytest.raises(ValueError):
        context.run(pairs, [(0, len(context.positions)+1)])
    with pytest.raises(ValueError):
        context.run(pairs, regions, tile_sites=0)
