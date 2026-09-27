"""Checkpoint boundaries must preserve the full forward/backward recurrence."""
from pathlib import Path
import numpy as np
import pytest
import gamma_smc_cu as g


@pytest.fixture
def context():
    rng = np.random.default_rng(8725521)
    G = rng.integers(0, 2, (12, 257), dtype=np.uint8)
    gaps = rng.integers(1, 80, 257)
    gaps[[0, 31, 64, 256]] = [217, 4000, 1, 777]
    positions = np.cumsum(gaps).astype(float)
    flow = str(Path(g.__file__).parent/'default_flow_field.txt')
    return g.FlowContext(G, positions, 10000., 1.25e-8, 1e-8, flow, 0)


@pytest.mark.parametrize("tile", [1, 31, 64, 256, 1000])
@pytest.mark.parametrize("ci,posterior", [(False, False), (True, False), (True, True)])
def test_checkpointed_exact_boundaries(context, tile, ci, posterior):
    pairs = [(i % 6, 6 + (i % 6)) for i in range(35)]
    expected = context.run_fb_blockwise(pairs, core_block_sites=257, flank_sites=0,
        pair_batch_size=35, max_streams=1, mean_only=not ci, return_posterior=posterior)
    actual = context.run_fb_checkpointed(pairs, tile_sites=tile,
        mean_only=not ci, return_posterior=posterior)
    for key in ('mean', 'lower', 'upper', 'posterior_alpha', 'posterior_beta'):
        if key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_checkpointed_invalid_and_empty(context):
    with pytest.raises(ValueError):
        context.run_fb_checkpointed([(0, 1)], tile_sites=0)
    with pytest.raises(ValueError):
        context.run_fb_checkpointed([(0, 12)], tile_sites=64)
    assert context.run_fb_checkpointed([], tile_sites=64)['mean'].shape == (257, 0)
    # A failed request must not poison the context.
    a = context.run_fb_checkpointed([(0, 1)], tile_sites=31)
    b = context.run_fb_checkpointed([(0, 1)], tile_sites=64)
    np.testing.assert_array_equal(a['mean'], b['mean'])
    assert a['checkpoint_workspace_bytes'] < 257 * 12


def test_checkpointed_rejects_nonfinite_calibrated_means(context):
    with pytest.raises(RuntimeError, match='finite and strictly positive'):
        context.run_fb_checkpointed([(0, 1)], tile_sites=31,
                                   calibration_factor=1e38)
    # Failure releases scratch and leaves a reusable context.
    result = context.run_fb_checkpointed([(0, 1)], tile_sites=31)
    assert result['posterior_means_validated']
    assert np.isfinite(result['mean']).all() and (result['mean'] > 0).all()


@pytest.mark.parametrize("tile", [1, 31, 128, 1000])
def test_region_moments_preserve_pair_distribution(context, tile):
    pairs = [(i % 6, 6 + (i % 6)) for i in range(35)]
    regions = [(0, 1), (0, 257), (30, 129), (128, 257), (63, 65)]
    factor = np.float32(1.590076987)
    reference = context.run_fb_blockwise(pairs, core_block_sites=257,
        flank_sites=0, pair_batch_size=35, max_streams=1, mean_only=True)
    means = reference['mean'] * factor
    actual = context.run_fb_region_moments(pairs, regions, tile_sites=tile,
                                         calibration_factor=float(factor))
    for i, (start, stop) in enumerate(regions):
        values = means[start:stop].astype(np.float64)
        linear = values.mean(axis=0)
        logarithmic = np.log(values).mean(axis=0)
        np.testing.assert_allclose(actual['region_mean'][i], linear, rtol=1e-12)
        np.testing.assert_allclose(actual['region_mean_log'][i], logarithmic, rtol=1e-12)
        edges = np.linspace(np.log(10), np.log(1e6), 51)
        np.testing.assert_array_equal(np.histogram(actual['region_mean_log'][i], edges)[0],
                                      np.histogram(logarithmic, edges)[0])
        np.testing.assert_array_equal(actual['region_mean'][i] < 1000, linear < 1000)
        np.testing.assert_array_equal(actual['region_mean_log'][i] < np.log(1000),
                                      logarithmic < np.log(1000))
    with pytest.raises(ValueError):
        context.run_fb_region_moments(pairs, [(1, 1)], tile_sites=tile)
