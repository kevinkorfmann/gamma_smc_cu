"""Direct NPY export must preserve every bit, file boundary, and failure signal."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import gamma_smc_cu as g


@pytest.fixture
def panel():
    rng = np.random.default_rng(28094)
    G = rng.integers(0, 2, (12, 257), dtype=np.uint8)
    G[:, [1, 63, 256]] = 0
    positions = np.cumsum(rng.integers(1, 4000, G.shape[1])).astype(float)
    pairs = [(i, j) for i in range(12) for j in range(i+1, 12)][:35]
    return G, positions, pairs


@pytest.mark.parametrize('devices,tile,buffers,workers', [([0], 1, 2, 1), ([0], 31, 3, 2),
                                                ([0], 1000, 2, 4), ([0, 1], 64, 3, 2)])
def test_exact_export(panel, tmp_path, devices, tile, buffers, workers):
    if len(devices) > g._core.get_device_count():
        pytest.skip('requires two GPUs')
    G, positions, pairs = panel
    kwargs = dict(mu=2.1e-8, rho=0.9e-8, physical_mu=1.25e-8, auto_estimate_theta=False)
    expected = g.infer_blockwise(G, positions, pairs=pairs, core_block_sites=1000,
                                flank_sites=0, mean_only=True, **kwargs)
    affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    out = tmp_path/'export'
    result = g.export_dense(G, positions, pairs=pairs, output_dir=out, pair_batch_size=13,
                           tile_sites=tile, staging_buffers=buffers, gpu_ids=devices,
                           workers_per_gpu=workers,
                           durable=True, **kwargs)
    assert result == json.loads((out/'manifest.json').read_text())
    assert result['complete']
    assert result['calibration'] == expected['metadata']
    np.testing.assert_array_equal(np.load(out/'positions.npy'), expected['positions'])
    np.testing.assert_array_equal(np.load(out/'pairs.npy'), pairs)
    actual = np.concatenate([np.load(out/r['file']) for r in result['files']], axis=1)
    np.testing.assert_array_equal(actual, expected['mean'])
    assert not list(out.glob('*.partial'))
    if affinity is not None:
        assert os.sched_getaffinity(0) == affinity
    with pytest.raises(FileExistsError):
        g.export_dense(G, positions, pairs=pairs, output_dir=out, **kwargs)


def test_empty_exports(panel, tmp_path):
    G, positions, pairs = panel
    empty = g.export_dense(G, positions, pairs=[], output_dir=tmp_path/'empty')
    assert empty['complete'] and empty['files'] == []
    mono = g.export_dense(np.zeros_like(G), positions, pairs=pairs,
                         output_dir=tmp_path/'mono', auto_estimate_theta=False)
    assert mono['shape'] == [0, len(pairs)]
    assert np.load(tmp_path/'mono'/mono['files'][0]['file']).shape == (0, len(pairs))


def test_native_payload_boundaries_and_failure(panel, tmp_path):
    G, positions, pairs = panel
    flow = str(Path(g.__file__).parent/'default_flow_field.txt')
    context = g.FlowContext(G, positions, 10000., 1.25e-8, 1e-8, flow, 0)
    expected = context.run_fb_checkpointed(pairs, tile_sites=31)['mean']
    path = tmp_path/'raw'
    prefix, suffix = b'HEADER' * 7, b'TAIL' * 9
    with path.open('w+b') as f:
        f.write(prefix + bytes(expected.nbytes) + suffix)
        f.flush()
        context.export_mean_fd(pairs, f.fileno(), len(prefix), tile_sites=31)
        f.seek(0)
        assert f.read(len(prefix)) == prefix
        np.testing.assert_array_equal(np.frombuffer(f.read(expected.nbytes), np.float32).reshape(expected.shape), expected)
        assert f.read() == suffix
        with pytest.raises(RuntimeError, match='finite and strictly positive'):
            context.export_mean_fd(pairs, f.fileno(), len(prefix), tile_sites=31,
                                   calibration_factor=1e38)
        # A GPU validation failure must drain outstanding I/O and leave reusable state.
        context.export_mean_fd(pairs, f.fileno(), len(prefix), tile_sites=31)
    with path.open('rb') as f:
        with pytest.raises(ValueError, match='writable'):
            context.export_mean_fd(pairs, f.fileno(), len(prefix))
    with path.open('ab') as f:
        with pytest.raises(ValueError, match='O_APPEND'):
            context.export_mean_fd(pairs, f.fileno(), len(prefix))


def test_failed_export_has_no_complete_manifest(panel, tmp_path, monkeypatch):
    import gamma_smc_cu.export as exporting
    G, positions, pairs = panel
    def fail(*args):
        raise OSError(28, 'simulated disk full')
    monkeypatch.setattr(exporting, '_preallocate_file', fail)
    out = tmp_path/'failed'
    with pytest.raises(OSError, match='disk full'):
        g.export_dense(G, positions, pairs=pairs, output_dir=out, pair_batch_size=10)
    assert not (out/'manifest.json').exists()
    assert not list(out.glob('*.partial'))


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux fallocate')
@pytest.mark.parametrize('error', [0, 95, 28, 9])
def test_preallocation_unsupported_and_real_errors(tmp_path, monkeypatch, error):
    import ctypes
    from types import SimpleNamespace
    from gamma_smc_cu.export import _preallocate_file

    def allocate(fd, mode, offset, size):
        ctypes.set_errno(error)
        return -1 if error else 0

    def unexpected_emulation(*args):
        raise AssertionError('Do not invoke glibc zero-write emulation')

    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(fallocate=allocate))
    monkeypatch.setattr(os, 'posix_fallocate', unexpected_emulation)
    with (tmp_path/'payload').open('xb') as f:
        f.write(b'HEADER'); f.flush()
        os.ftruncate(f.fileno(), 8192)
        if error in (0, 95):  # Unsupported filesystems keep the sparse sizing.
            _preallocate_file(f.fileno(), 8192)
        else:
            with pytest.raises(OSError) as caught:
                _preallocate_file(f.fileno(), 8192)
            assert caught.value.errno == error
    data = (tmp_path/'payload').read_bytes()
    assert data == b'HEADER' + bytes(8192-6)
