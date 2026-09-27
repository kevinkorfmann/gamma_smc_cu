"""Scientific input identity checks; no GPU is needed."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_matched_panel import check_panel, pair_set, parse_cpu_timers, prepare
import run_matched_panel as matched


def test_unique_pair_sets_cover_same_full_cohort():
    for specification, n_expected in [('within',178), ('cross:1',708), ('cross:15',9780), ('cross:89',31684), ('all',63190)]:
        pairs, first, second = pair_set(356, specification)
        assert len(pairs) == n_expected
        assert len({frozenset(pair) for pair in pairs}) == n_expected
        assert set(pairs.ravel()) == set(range(356))
        assert set(first) | set(second) == set(range(178))
        assert not set(first) & set(second)
    pairs, _, _ = pair_set(6, 'cross:1')
    # Actual upstream -S complement, -T subset ordering, including orientation.
    np.testing.assert_array_equal(pairs, [(2,0),(2,1),(3,0),(3,1),(4,0),(4,1),(5,0),(5,1)])
    with pytest.raises(ValueError):
        pair_set(356, 'cross:178')


@pytest.mark.skipif(not shutil.which('bgzip') or not shutil.which('tabix'), reason='requires standard VCF compression tools')
@pytest.mark.parametrize('cache_kind', ['npz', 'memmap'])
def test_all_input_formats_share_once_filtered_full_panel(tmp_path, cache_kind):
    # Site 2 is polymorphic only outside YRI; site 3 is all alternate within
    # YRI. Both must be excluded once, regardless of selected inference pairs.
    G = np.array([[0,0,0,1,0], [1,0,0,1,0], [0,1,0,1,0], [0,0,0,1,0],
                  [0,0,0,1,1], [0,0,0,1,0], [0,0,1,0,0], [0,0,0,0,0]], dtype=np.int8)
    source = tmp_path/'chr22.npz'
    np.savez(source, G=G, positions=[1,11,21,31,41], sample_ids=['a','b','c','d'])
    samples = tmp_path/'samples.txt'
    samples.write_text('x a x x x YRI x\nx b x x x YRI x\nx c x x x YRI x\nx d x x x CEU x\n')
    output = tmp_path/'panel'
    output.mkdir()
    cache_dir = None
    if cache_kind == 'memmap':
        from analysis.rerun.cache import prepare as prepare_cache
        cache_dir = tmp_path/'cache'
        prepare_cache(source, cache_dir/'parsed/chr22')
    info = prepare(argparse.Namespace(parsed_npz=str(source), cache_dir=cache_dir, samples_file=str(samples), positions_base=1,
                                     population='YRI', chromosome='22', mu=1.25e-8, rho=1e-8), output)
    assert info['n_haplotypes'] == 6 and info['n_sites'] == 3
    np.testing.assert_array_equal(np.load(output/'positions.npy'), [0,10,40])
    expected = G[:6, [0,1,4]]
    np.testing.assert_array_equal(np.load(output/'G.npy'), expected)
    with gzip.open(output/'panel.vcf.gz','rt') as stream:
        lines = [line.split() for line in stream if not line.startswith('#')]
    np.testing.assert_array_equal([int(line[1]) for line in lines], [1,11,41])
    actual_vcf = np.asarray([[int(allele) for gt in row[9:] for allele in gt.split('|')] for row in lines]).T
    np.testing.assert_array_equal(actual_vcf, expected)
    with gzip.open(output/'asmc.hap.gz','rt') as stream:
        rows = [line.split() for line in stream]
    np.testing.assert_array_equal([int(row[2]) for row in rows], [1,11,41])
    np.testing.assert_array_equal(np.asarray([row[5:] for row in rows], dtype=int).T, expected)
    assert info['theta'] == pytest.approx(1/41)
    assert check_panel(output)['n_sites'] == 3
    (output/'sample_ids.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum mismatch'):
        check_panel(output)


def test_cpu_workload_log_cannot_silently_change_markers_or_pair_count():
    stdout = '\x1b[1mRead 178 samples.\x1b[0m\nRead 400000 segregating sites.\nApplying to 708 haplotype pairs\n'
    stdout += 'Emissions preparation time:\t1.123 secs\nForward pass time:\t2.000 secs\nBackward pass time:\t3.000 secs\nOutput time:\t4.000 secs\n'
    timers = parse_cpu_timers(stdout, 356, 400000, 708)
    assert timers['decode_seconds'] == pytest.approx(6.123)
    assert timers['output_seconds'] == 4
    with pytest.raises(ValueError, match='workload mismatch'):
        parse_cpu_timers(stdout, 356, 399999, 708)
    with pytest.raises(ValueError, match='workload mismatch'):
        parse_cpu_timers(stdout, 356, 400000, 709)


def test_demography_rescaling_preserves_mutation_scaled_times_and_haploid_sizes():
    from prepare_asmc_decoding import rescale_demography, SOURCE_MU
    history=np.array([[0.,10000.],[100.,20000.],[1000.,30000.]])
    target_mu=1.25e-8
    corrected=rescale_demography(history,target_mu)
    np.testing.assert_allclose(corrected*target_mu,history*SOURCE_MU,rtol=1e-15)
    assert corrected[1,0]==132 and corrected[0,1]==13200
    np.testing.assert_array_equal(history[:,0],[0,100,1000])
    with pytest.raises(ValueError):
        rescale_demography(np.array([[0.,10000.],[0.,20000.]]),target_mu)


@pytest.mark.parametrize('workers', [1, 2, 3, 8])
def test_pair_shards_are_exact_once_without_changing_orientation(workers):
    pairs, _, _ = pair_set(10, 'cross:2')
    shards = matched.shard_pairs(pairs, workers, 10)
    np.testing.assert_array_equal(np.concatenate(shards), pairs)
    assert len(shards) == workers and min(map(len, shards)) > 0
    assert max(map(len, shards))-min(map(len, shards)) <= 1
    assert len({frozenset(pair) for shard in shards for pair in shard}) == len(pairs)
    for bad in [np.array([[0, 1], [1, 0]]), np.array([[0, 0]]), np.array([[0, 10]]),
                np.array([[0., 1.]]), np.array([0, 1])]:
        with pytest.raises(ValueError):
            matched.shard_pairs(bad, 1, 10)
    with pytest.raises(ValueError, match='exceed pair count'):
        matched.shard_pairs(pairs, len(pairs)+1, 10)


def test_affinity_uses_allowed_physical_cores_not_smt_siblings():
    topology = '# CPU,Core,Socket\n0,0,0\n1,1,0\n2,0,1\n3,1,1\n4,0,0\n5,1,0\n6,0,1\n7,1,1\n'
    assert matched.select_physical_cpus(topology, {4, 5, 6, 7}, 4) == [4, 5, 6, 7]
    assert matched.select_physical_cpus(topology, {0, 4, 1, 5}, 2) == [0, 1]
    with pytest.raises(ValueError, match='insufficient allocated physical cores'):
        matched.select_physical_cpus(topology, {0, 4, 1, 5}, 3)
    with pytest.raises(ValueError, match='does not describe'):
        matched.select_physical_cpus(topology, {99}, 1)


def test_affinity_plan_rejects_worker_count_above_slurm_allocation(monkeypatch):
    monkeypatch.setattr(matched.os, 'sched_getaffinity', lambda _: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(matched.os, 'sched_setaffinity', lambda *_: None, raising=False)
    monkeypatch.setattr(matched.subprocess, 'check_output', lambda *a, **k: '0,0,0\n1,1,0\n2,2,0\n3,3,0\n')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    with pytest.raises(ValueError, match='SLURM_CPUS_PER_TASK'):
        matched.asmc_affinity_plan(3)


@pytest.mark.parametrize('problem', ['shape', 'dtype', 'reversed', 'nan', 'negative'])
def test_native_output_validation_rejects_changed_workload_or_invalid_values(problem):
    pairs = np.array([[0, 1], [2, 3]])
    values = np.ones((2, 3), dtype=np.float32)
    returned = pairs.copy()
    if problem == 'shape':
        values = values[:, :2]
    elif problem == 'dtype':
        values = values.astype(np.float64)
    elif problem == 'reversed':
        returned = returned[:, ::-1]
    else:
        values[1, 1] = np.nan if problem == 'nan' else -1
    with pytest.raises((ValueError, AssertionError)):
        matched.validate_asmc_means(values, returned, pairs, 3)


def test_parallel_worker_stage_sum_is_never_reported_as_wall_time():
    workers = []
    for index in range(2):
        workers.append(dict(worker_index=index, native_total_seconds=10., setup_seconds=1.,
                            native_decode_seconds=6., extraction_seconds=.2, decode_seconds=6.2,
                            validation_seconds=.3, output_seconds=1., cpu_work_seconds=7.7,
                            native_decode_cpu_seconds=5.9, actual_params={'batchSize': 32},
                            native_states=51, provenance={'binary': 'abc'}))
    result = matched.summarize_asmc_workers(workers, 11.5, {})
    assert result['native_total_seconds'] == 11.5
    assert result['decode_seconds'] is None and result['native_decode_seconds'] is None
    assert result['extraction_seconds'] is None and result['output_seconds'] is None
    assert result['worker_stage_wall_sums']['decode_seconds'] == 12.4
    assert result['cpu_work_seconds'] == 15.4
    assert matched.optional_timer_distribution([result], 'decode_seconds') is None
    single = matched.summarize_asmc_workers(workers[:1], 11.5, {})
    assert single['native_total_seconds'] == 10. and single['decode_seconds'] == 6.2
    with pytest.raises(ValueError, match='inconsistent timing'):
        matched.optional_timer_distribution([single, result], 'decode_seconds')
    workers[1]['native_states'] = 50
    with pytest.raises(ValueError, match='native_states'):
        matched.summarize_asmc_workers(workers, 11.5, {})


FAKE_ASMC = '''
import json
from pathlib import Path
import weakref
import numpy as np

class DecodingParams:
    def __init__(self, **k):
        self.root = Path(k['in_file_root']).parent
        self.batchSize = 64
        self.noBatches = k['no_batches']
        self.decodingSequence = k['decoding_mode_string'] == 'sequence'
        self.usingCSFS = k['using_CSFS']
        self.compress = k['compress']
        self.skipCSFSdistance = k['skip_CSFS_distance']
        self.doPosteriorSums = k['do_posterior_sums']
        self.doMajorMinorPosteriorSums = k['do_major_minor_posterior_sums']
        self.doPerPairMAP = k['do_per_pair_MAP']
        self.jobs = k.get('jobs', 1)
        self.jobInd = k.get('job_ind', 1)
        self.useKnownSeed = False

class Result:
    def __init__(self, matrix, pairs):
        self.per_pair_posterior_means = matrix.view()
        self.per_pair_indices = [(a, str(a), b, str(b)) for a, b in pairs]

class ASMC:
    def __init__(self, params):
        assert isinstance(params, DecodingParams)
        self.params = params
        self.info = json.loads((params.root/'panel.json').read_text())
        self.positions = np.load(params.root/'positions.npy')
        self.loan = self.matrix = None
    def set_store_per_pair_posterior_mean(self, flag):
        assert flag
    def get_decoding_params(self):
        return self.params
    def get_haploid_sample_size(self):
        return self.info['n_haplotypes']
    def get_num_sites(self):
        return len(self.positions)
    def get_physical_positions(self):
        return self.positions + 1
    def get_expected_times(self):
        return [1., 10., 100.]
    def decode_pairs(self, a, b):
        assert self.loan is None or self.loan() is None, 'borrowed result survived next decode'
        if self.matrix is not None:
            self.matrix.fill(-99)
        self.pairs = list(zip(a, b))
        self.matrix = np.asarray([self.positions + 1 + i*10 + j for i, j in self.pairs], dtype=np.float32)
    def get_ref_of_results(self):
        result = Result(self.matrix, self.pairs)
        self.loan = weakref.ref(result)
        return result
    def get_copy_of_results(self):
        raise AssertionError('production path must borrow results')
'''


@pytest.fixture
def fake_asmc_jobs(tmp_path, monkeypatch):
    package = tmp_path/'asmc'; package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'asmc.py').write_text(FAKE_ASMC)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ['asmc', 'asmc.asmc']:
        monkeypatch.delitem(sys.modules, name, raising=False)
    panel = tmp_path/'panel'; panel.mkdir()
    positions = np.array([0, 9, 29])
    np.save(panel/'positions.npy', positions)
    info = dict(n_haplotypes=10, n_sites=3)
    (panel/'panel.json').write_text(json.dumps(info))
    pairs, _, _ = pair_set(10, 'cross:2')
    jobs, offset = [], 0
    for index, shard in enumerate(matched.shard_pairs(pairs, 2, 10)):
        folder = tmp_path/f'worker{index}'; folder.mkdir()
        np.save(folder/'pairs.npy', shard)
        jobs.append(dict(worker_index=index, pair_start=offset, cpu=None, panel=str(panel), info=info,
                         directory=str(folder), pairs_file=str(folder/'pairs.npy'),
                         dq=str(tmp_path/'dq'), outer_batch=3, native_batch=32))
        offset += len(shard)
    yield jobs, pairs, positions
    for name in ['asmc', 'asmc.asmc']:
        sys.modules.pop(name, None)


def test_borrowed_outputs_saved_before_reuse_and_full_cohort_retained(fake_asmc_jobs):
    jobs, pairs, positions = fake_asmc_jobs
    records = [matched.asmc_worker(job) for job in jobs]
    restored = np.concatenate([np.concatenate([np.load(path).T for path in sorted(Path(job['directory']).glob('mean_batch*.npy'))]) for job in jobs])
    expected = np.asarray([positions+1+a*10+b for a, b in pairs], dtype=np.float32)
    np.testing.assert_array_equal(restored, expected)
    for record in records:
        assert record['actual_params']['usingCSFS']
        assert not record['actual_params']['doPosteriorSums']
        assert not record['actual_params']['doMajorMinorPosteriorSums']
        assert record['actual_params']['batchSize'] == 32
        assert record['actual_params']['useKnownSeed'] is True
        assert record['actual_params']['fixed_upstream_seed'] == 1234
        assert record['actual_params']['jobs'] == record['actual_params']['jobInd'] == 1
        assert record['native_states'] == 3
        assert record['provenance']['module_files_sha256']
        assert sum(record['outer_call_pair_counts']) == record['n_pairs']
        assert record['decode_seconds'] == record['native_decode_seconds']+record['extraction_seconds']


def test_spawned_pair_shards_run_in_distinct_processes(fake_asmc_jobs, monkeypatch):
    jobs, pairs, _ = fake_asmc_jobs
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    records = matched.spawned_asmc_workers(jobs, 60)
    assert len({r['pid'] for r in records}) == 2
    assert sum(r['n_pairs'] for r in records) == len(pairs)
    assert [r['pair_start'] for r in records] == [job['pair_start'] for job in jobs]
    assert matched.os.environ['OMP_NUM_THREADS'] == '7'
    assert matched.summarize_asmc_workers(records, 10., {})['decode_seconds'] is None


def test_worker_rejects_cpu_outside_inherited_affinity(fake_asmc_jobs, monkeypatch):
    jobs, _, _ = fake_asmc_jobs
    job = {**jobs[0], 'cpu': 99}
    monkeypatch.setattr(matched.os, 'sched_getaffinity', lambda _: {2, 3}, raising=False)
    monkeypatch.setattr(matched.os, 'sched_setaffinity', lambda *_: pytest.fail('must reject before pinning'), raising=False)
    with pytest.raises(ValueError, match='outside inherited allocation'):
        matched.asmc_worker(job)


def test_worker_restores_affinity_when_full_cohort_validation_fails(fake_asmc_jobs, monkeypatch):
    jobs, _, _ = fake_asmc_jobs
    affinity, calls = {2, 3}, []
    def set_affinity(_, selected):
        affinity.clear()
        affinity.update(selected)
        calls.append(set(selected))
    monkeypatch.setattr(matched.os, 'sched_getaffinity', lambda _: set(affinity), raising=False)
    monkeypatch.setattr(matched.os, 'sched_setaffinity', set_affinity, raising=False)
    job = {**jobs[0], 'cpu': 2, 'info': {**jobs[0]['info'], 'n_haplotypes': 12}}
    with pytest.raises(ValueError, match='different cohort'):
        matched.asmc_worker(job)
    assert calls == [{2}, {2, 3}] and affinity == {2, 3}


def test_spawned_worker_failure_cannot_return_success_or_leave_environment_changed(fake_asmc_jobs, monkeypatch):
    jobs, _, _ = fake_asmc_jobs
    jobs[1] = {**jobs[1], 'info': {**jobs[1]['info'], 'n_haplotypes': 12}}
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '5')
    with pytest.raises(RuntimeError, match='worker'):
        matched.spawned_asmc_workers(jobs, 60)
    assert matched.os.environ['OPENBLAS_NUM_THREADS'] == '5'
