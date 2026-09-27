"""Tuning must preserve the scientific workload and consume native views safely."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tune_asmc as tuning


def fake_module(positions, n_haplotypes=8, multiplier=1.0):
    class Params:
        def __init__(self, **kwargs):
            self.batchSize = 64
            self.useKnownSeed = False
            self.jobs = kwargs.get('jobs', 1)
            self.jobInd = kwargs.get('job_ind', 1)
            self.noBatches = kwargs.get('no_batches', False)
            self.decodingSequence = kwargs.get('decoding_mode_string') == 'sequence'
            self.usingCSFS = kwargs.get('using_CSFS', True)
            self.compress = kwargs.get('compress', False)
            self.skipCSFSdistance = float('inf') if self.compress else kwargs.get('skip_CSFS_distance', 0)
            self.doPosteriorSums = kwargs.get('do_posterior_sums', False)
            self.doMajorMinorPosteriorSums = kwargs.get('do_major_minor_posterior_sums', False)
            self.doPerPairMAP = kwargs.get('do_per_pair_MAP', False)

    class Result:
        def __init__(self, matrix, pairs):
            self.per_pair_posterior_means = matrix
            self.per_pair_indices = [(a, str(a), b, str(b)) for a, b in pairs]

    class ASMC:
        instances = []

        def __init__(self, first, *args, **kwargs):
            if isinstance(first, Params):
                self.params = first
                assert first.useKnownSeed is True, 'Seed must be set before native construction'
            else:
                self.params = Params(decoding_mode_string='sequence', do_posterior_sums=True,
                                     do_major_minor_posterior_sums=True, do_per_pair_MAP=True)
            self.loan = None
            self.matrix = None
            self.calls = []
            ASMC.instances.append(self)

        def set_store_per_pair_posterior_mean(self, flag):
            assert flag is True

        def get_decoding_params(self):
            return self.params

        def get_haploid_sample_size(self):
            return n_haplotypes

        def get_num_sites(self):
            return len(positions)

        def get_physical_positions(self):
            return np.asarray(positions) + 1

        def get_expected_times(self):
            return [1., 2., 3.]

        def decode_pairs(self, a, b):
            # Returned native views cannot be held across decode calls.
            assert self.loan is None or self.loan() is None
            if self.matrix is not None:
                self.matrix.fill(-99)
            self.calls.append(list(zip(a, b)))
            self.pairs = list(zip(a, b))
            self.matrix = np.asarray([(1 + np.asarray(positions) + i * .5 + j) * multiplier
                                      for i, j in self.pairs], dtype=np.float32)

        def get_ref_of_results(self):
            result = Result(self.matrix.view(), self.pairs)
            self.loan = weakref.ref(result)
            return result

        def get_copy_of_results(self):
            return Result(self.matrix.copy(), self.pairs)

    return SimpleNamespace(ASMC=ASMC, DecodingParams=Params)


def specification(tmp_path):
    panel = tmp_path / 'panel'; panel.mkdir()
    positions = np.asarray([0, 9, 21, 100, 999], dtype=np.int64)
    np.save(panel / 'positions.npy', positions)
    pairs = tuning.select_pairs(8, 3)
    np.save(tmp_path / 'pairs.npy', pairs)
    return dict(panel_dir=str(panel), decoding_quantities=str(tmp_path / 'dq'),
                panel=dict(n_haplotypes=8, n_sites=len(positions)),
                pairs_file=str(tmp_path / 'pairs.npy'), repeats=2,
                rtol=1e-6, atol=1e-5, retain_predictions=True), positions, pairs


def test_profiles_disable_unused_work_without_changing_model(tmp_path):
    module = fake_module([0, 1])
    profiles = tuning.profiles([8, 32, 512], include_compress=True)
    for profile in profiles:
        obj, actual = tuning.create_asmc(module, profile, tmp_path, tmp_path / 'dq', tmp_path)
        assert actual['batchSize'] == profile['native_batch'] and not actual['noBatches']
        assert actual['useKnownSeed'] and actual['fixed_upstream_seed'] == 1234
        assert actual['jobs'] == actual['jobInd'] == 1
        if profile['constructor'] == 'reconstructed_defaults':
            assert actual['doPosteriorSums'] and actual['doMajorMinorPosteriorSums']
        else:
            assert not actual['doPosteriorSums'] and not actual['doMajorMinorPosteriorSums']
        if not profile['changed_model']:
            assert actual['usingCSFS'] and not actual['compress'] and actual['skipCSFSdistance'] == 0
        else:
            assert actual['compress'] and not actual['usingCSFS'] and actual['skipCSFSdistance'] == 'inf'


def test_distinct_pairs_do_not_thin_cohort_or_allow_reverse_duplicates():
    pairs = tuning.select_pairs(356, 128)
    assert pairs.shape == (128, 2)
    assert len(set(map(tuple, pairs))) == 128
    assert pairs[0].tolist() == [0, 1] and pairs[-1].tolist() == [354, 355]
    with pytest.raises(ValueError, match='distinct'):
        tuning.select_pairs(356, 2, [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match='integer'):
        tuning.select_pairs(356, 1, np.asarray([[0., 1.]]))


def test_full_panel_reference_survives_native_buffer_reuse(tmp_path):
    spec, positions, pairs = specification(tmp_path)
    baseline_dir = tmp_path / 'baseline'; baseline_dir.mkdir()
    baseline = tuning.run_profile(spec, tuning.profiles([1])[0], baseline_dir,
                                   module=fake_module(positions))
    ref = baseline['reference_path']
    for profile in tuning.profiles([1, 32])[1:]:
        output = tmp_path / profile['name']; output.mkdir()
        module = fake_module(positions)
        result = tuning.run_profile(spec, profile, output, ref, module=module)
        assert result['eligible_for_main_comparison']
        assert result['status'] == 'ok'
        for record in result['repetitions']:
            assert all(check['exactly_equal'] for check in record['comparisons'])
            assert sum(record['outer_call_pair_counts']) == len(pairs)
            for key in ['setup_seconds', 'native_decode_seconds', 'extraction_seconds',
                        'validation_seconds', 'output_seconds', 'total_seconds']:
                assert record[key] >= 0
        for instance in module.ASMC.instances:
            assert [pair for batch in instance.calls for pair in batch] == list(map(tuple, pairs))
        np.testing.assert_array_equal(np.load(output / 'rep00/posterior_means.npy'), np.load(ref))
    np.testing.assert_array_equal(np.load(Path(spec['panel_dir']) / 'positions.npy'), positions)


@pytest.mark.parametrize('invalid', [0., -1., np.nan, np.inf])
def test_invalid_posterior_is_never_hidden(invalid):
    values = np.ones((2, 5), dtype=np.float32); values[1, 3] = invalid
    with pytest.raises(ValueError, match='finite and strictly positive'):
        tuning.comparison(values, None, 1e-6, 1e-5)


def test_parity_failure_cannot_become_main_timing_winner(tmp_path):
    spec, positions, _ = specification(tmp_path)
    baseline_dir = tmp_path / 'baseline'; baseline_dir.mkdir()
    baseline = tuning.run_profile(spec, tuning.profiles([32])[0], baseline_dir,
                                   module=fake_module(positions))
    profile = next(p for p in tuning.profiles([32]) if p['constructor'] == 'explicit')
    output = tmp_path / 'incorrect'; output.mkdir()
    result = tuning.run_profile(spec, profile, output, baseline['reference_path'],
                                module=fake_module(positions, multiplier=1.01))
    assert result['status'] == 'parity_failed'
    assert not result['eligible_for_main_comparison']
    assert sum(c['n_outside_tolerance'] for r in result['repetitions'] for c in r['comparisons']) > 0
    profile = tuning.profiles([32], include_compress=True)[-1]
    output = tmp_path / 'changed_model'; output.mkdir()
    result = tuning.run_profile(spec, profile, output, baseline['reference_path'],
                                module=fake_module(positions, multiplier=1.01))
    assert result['status'] == 'ok' and not result['eligible_for_main_comparison']


def test_manifest_checks_reject_mutation_before_any_decoding(tmp_path):
    panel = tmp_path / 'panel'; panel.mkdir()
    (panel / 'panel.json').write_text(json.dumps(dict(positions_base=0)))
    np.save(panel / 'positions.npy', [0, 99])
    tuning.write_json(panel / 'manifest.json', {p.name:{'sha256':tuning.sha256(p)} for p in panel.iterdir()})
    assert tuning.check_panel(panel)['positions_base'] == 0
    np.save(panel / 'positions.npy', [0, 100])
    with pytest.raises(ValueError, match='checksum'):
        tuning.check_panel(panel)


def test_default_retention_keeps_reference_and_hashes_all_timed_outputs(tmp_path):
    spec, positions, _ = specification(tmp_path)
    spec['retain_predictions'] = False
    output = tmp_path / 'baseline'; output.mkdir()
    result = tuning.run_profile(spec, tuning.profiles([32])[0], output, module=fake_module(positions))
    assert Path(result['reference_path']).exists()
    assert not (output / 'rep00/posterior_means.npy').exists()
    assert len({r['predictions_sha256'] for r in result['repetitions']}) == 1
    assert all(r['predictions_bytes'] > 0 for r in result['repetitions'])
