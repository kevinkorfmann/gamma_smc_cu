"""The worker gate must compare every saved value and fail closed on provenance."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_asmc_workers as gate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def refresh_manifest(root):
    write_json(root/'manifest.json', {str(path.relative_to(root)): {
        'sha256': gate.sha256(path), 'bytes': path.stat().st_size}
        for path in root.rglob('*') if path.is_file() and path.name != 'manifest.json'})


def make_run(root, workers, outer_batch):
    root.mkdir()
    ns, nh = 9001, 12
    pairs = np.arange(nh, dtype=np.int32).reshape(-1, 2)
    np.save(root/'pairs.npy', pairs)
    params = dict(useKnownSeed=True, fixed_upstream_seed=1234, jobs=1, jobInd=1,
                  decodingSequence=True, usingCSFS=True, compress=False, skipCSFSdistance=0,
                  noBatches=False, doPosteriorSums=False, doMajorMinorPosteriorSums=False,
                  doPerPairMAP=False, batchSize=32)
    records, offset = [], 0
    for index, shard in enumerate(np.array_split(pairs, workers)):
        folder = root/f'rep00/worker{index:03d}'
        folder.mkdir(parents=True)
        np.save(folder/'pairs.npy', shard)
        np.save(folder/'expected_coalescent_times.npy', [1., 10., 100.])
        sizes = []
        for batch, lo in enumerate(range(0, len(shard), outer_batch)):
            subset = shard[lo:lo+outer_batch]
            values = np.asarray([np.arange(ns)+1+a*10+b for a, b in subset], dtype=np.float32).T
            np.save(folder/f'mean_batch{batch:05d}.npy', values)
            sizes.append(len(subset))
        records.append(dict(worker_index=index, pair_start=offset, n_pairs=len(shard),
                            pairs_sha256=gate.sha256(folder/'pairs.npy'), actual_params=params,
                            provenance={'asmc_version': '1.4.0', 'module_files_sha256': {'native.so': 'b'*64}},
                            native_states=3, outer_call_pair_counts=sizes))
        offset += len(shard)
    record = dict(repetition=0, warmup=False, n_workers=workers, workers=records,
                  actual_params=params, native_states=3)
    write_json(root/'rep00/timing.json', record)
    result = dict(status='ok', method='asmc', is_extrapolated=False, pair_set='within',
                  panel_manifest_sha256='a'*64, pairs_sha256=gate.sha256(root/'pairs.npy'),
                  panel={'n_haplotypes': nh, 'n_sites': ns}, n_distinct_pairs=len(pairs),
                  arguments={'asmc_workers': workers}, repetitions=[record])
    write_json(root/'result.json', result)
    write_json(root/'environment.json', {'external_files': {'decoding_quantities': {'sha256': 'd'*64}}})
    refresh_manifest(root)
    return root


@pytest.fixture
def runs(tmp_path):
    return make_run(tmp_path/'one', 1, 4), make_run(tmp_path/'many', 3, 1)


def change_record(root, mutate):
    result = json.loads((root/'result.json').read_text())
    mutate(result['repetitions'][0])
    write_json(root/'result.json', result)
    write_json(root/'rep00/timing.json', result['repetitions'][0])
    refresh_manifest(root)


def test_exact_comparison_across_different_worker_and_batch_boundaries(runs, tmp_path, monkeypatch):
    original = gate.np.asarray
    blocks = []
    def bounded_asarray(value, *args, **kwargs):
        if kwargs.get('dtype') == np.float64:
            blocks.append(value.shape)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(gate.np, 'asarray', bounded_asarray)
    report = gate.validate(*runs, tmp_path/'gate')
    assert report['passed'] and report['comparison']['exactly_equal']
    assert report['comparison']['n_values'] == 9001*6
    assert report['comparison']['maximum_absolute_difference'] == 0
    assert report['input_provenance']['parallel']['n_workers'] == 3
    assert all(sites <= 4096 and pairs <= 32 for sites, pairs in blocks)
    assert json.loads((tmp_path/'gate/validation.json').read_text())['passed']
    with pytest.raises(FileExistsError):
        gate.validate(*runs, tmp_path/'gate')


@pytest.mark.parametrize('value', [np.nan, np.inf, 0., -1., 1e7])
def test_every_value_including_last_batch_last_site_is_checked(runs, tmp_path, value):
    path = runs[1]/'rep00/worker002/mean_batch00001.npy'
    values = np.load(path)
    values[-1, -1] = value
    np.save(path, values)
    refresh_manifest(runs[1])
    report = gate.validate(*runs, tmp_path/'gate')
    assert not report['passed'] and report['errors']
    if np.isfinite(value) and value > 0:
        assert report['comparison']['n_outside_tolerance'] == 1
        assert not report['comparison']['exactly_equal']


def test_tolerated_roundoff_is_not_mislabeled_exact_equality(runs, tmp_path):
    path = runs[1]/'rep00/worker000/mean_batch00000.npy'
    values = np.load(path)
    values[0, 0] = np.nextafter(values[0, 0], np.float32(np.inf))
    np.save(path, values)
    refresh_manifest(runs[1])
    report = gate.validate(*runs, tmp_path/'gate')
    assert report['passed'] and not report['comparison']['exactly_equal']
    assert report['comparison']['maximum_absolute_difference'] > 0


@pytest.mark.parametrize('problem', ['dq', 'grid', 'seed', 'csfs', 'shard', 'missing', 'shape', 'checksum', 'status', 'panel'])
def test_provenance_and_incomplete_outputs_cannot_pass(runs, tmp_path, problem):
    root = runs[1]
    if problem == 'dq':
        write_json(root/'environment.json', {'external_files': {'decoding_quantities': {'sha256': 'e'*64}}})
    elif problem == 'grid':
        np.save(root/'rep00/worker002/expected_coalescent_times.npy', [1., 11., 100.])
    elif problem in ['seed', 'csfs']:
        key = 'fixed_upstream_seed' if problem == 'seed' else 'usingCSFS'
        change_record(root, lambda r: r['workers'][1]['actual_params'].update({key: 0}))
    elif problem == 'shard':
        change_record(root, lambda r: r['workers'][1].update(pair_start=0))
    elif problem == 'missing':
        (root/'rep00/worker002/mean_batch00001.npy').unlink()
    elif problem in ['shape', 'checksum']:
        np.save(root/'rep00/worker000/mean_batch00000.npy', np.ones((9000, 1), dtype=np.float32))
    else:
        result = json.loads((root/'result.json').read_text())
        result['status' if problem == 'status' else 'panel_manifest_sha256'] = 'incorrect'
        write_json(root/'result.json', result)
    if problem != 'checksum':
        refresh_manifest(root)
    report = gate.validate(*runs, tmp_path/'gate')
    assert not report['passed'] and report['errors']


def test_cli_failure_exits_nonzero_and_keeps_inspection_artifact(runs, tmp_path):
    # Reversed roles must fail even when every posterior agrees.
    output = tmp_path/'gate'
    command = [sys.executable, str(Path(gate.__file__)), '--single', str(runs[1]),
               '--parallel', str(runs[0]), '--output-dir', str(output)]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 1
    report = json.loads((output/'validation.json').read_text())
    assert not report['passed'] and 'single worker' in report['errors'][0]
