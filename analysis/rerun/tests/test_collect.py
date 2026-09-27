from pathlib import Path
import copy
import importlib.util
import json
import sys
import shutil

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1] / 'collect.py'
spec = importlib.util.spec_from_file_location('collect_rerun', FILE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


def hosts_and_gate(tmp_path):
    hosts = {name: tmp_path / name for name in ('betty', 'sesame')}
    for root in hosts.values():
        (root/'source').mkdir(parents=True)
        (root/'source/infer.py').write_text('frozen inference implementation')
        write_json(root / 'source/SOURCE_MANIFEST.json', {'git_commit': 'a'*40, 'files': {'infer.py': c.sha(root/'source/infer.py')}})
    gate = tmp_path / 'parity.json'
    write_json(gate, {'passed': True, 'source_manifest_sha256': c.sha(hosts['betty'] / 'source/SOURCE_MANIFEST.json')})
    return hosts, gate


def create_gene(root, chrom=22, pop='ASW', epoch=100, mu=1.25e-8, vcf_sha='v' * 64, value=100):
    tid = f'genome_chr{chrom}_{pop}'
    out = root / 'outputs/tasks' / tid / 'attempt-001'
    folder = out / 'results' / f'chr{chrom}'
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f'{pop}.npz', gene_id=np.array(['id']), gene_name=np.array(['GENE']),
             start=np.array([10]), end=np.array([20]), count=np.array([6]), n_pairs_total=6,
             n_sites_per_gene=np.array([2]), log_sum=np.array([6 * np.log(value)]))
    (folder / f'{pop}.csv').write_text(f'gene_name,tmrca\nGENE,{value}\n')
    cache_record = dict(schema=2, dtype='uint8', positions_dtype='int64', positions_base=1,
                        coordinate_convention='VCF POS one-based', shape=[4, 2], source_sha256=vcf_sha,
                        filters=dict(chromosome=str(chrom), rule='strict'),
                        members={n: {'sha256': n * 3} for n in ('G.npy', 'positions.npy', 'sample_ids.npy')})
    metadata = dict(complete=True, calibration=dict(time_units='generations', calibration='heterozygosity', denominator_convention='last position + 1'),
                    identity=dict(chromosome=chrom, population=pop, configuration=dict(
                        cache_dir=str(root / 'inputcache'), samples=str(root / 'samples'), output_dir=str(out),
                        chr=chrom, mu=mu, rho=1e-8, core_block_sites=65536, flank_sites=8192, pair_chunk=512,
                        sequence_length=None, resume=False),
                        source_sha256={'infer.py': c.sha(root/'source/infer.py')}, inputs=dict(cache=dict(format='extracted_npy', record=cache_record),
                            samples={'sha256': 'samples'}, genes={'sha256': f'genes{chrom}'})),
                    output_sha256={ext: c.sha(folder / (pop + ext)) for ext in ('.npz', '.csv')})
    write_json(folder / f'{pop}.metadata.json', metadata)
    outputs = {str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=c.sha(path)) for path in folder.iterdir()}
    done = dict(task_id=tid, source_manifest_sha256=c.sha(root / 'source/SOURCE_MANIFEST.json'), returncode=0,
                completed_epoch=epoch, output=f'/remote/host/run/outputs/tasks/{tid}/attempt-001', outputs=outputs)
    write_json(root / 'state' / (tid + '.done.json'), done)
    return folder


def test_first_completion_incremental_winners_and_572_row_report(tmp_path):
    hosts, gate = hosts_and_gate(tmp_path)
    create_gene(hosts['betty'], epoch=100, value=101)
    create_gene(hosts['sesame'], epoch=99, value=100)
    selected = tmp_path / 'selected'
    result = c.collect(hosts, selected, gate)
    assert result == dict(selected=1, expected=572, complete=False, newly_selected=1, postprocessing_ready=False,
                         validation_mode='cross_host', cross_host_parity_validated=True)
    registry = c.read_json(selected / 'winner_registry.json')
    assert registry['winners']['genome_chr22_ASW']['host'] == 'sesame'
    assert len((selected / 'inspection.tsv').read_text().splitlines()) == 573
    old_hash = c.sha(selected / 'genome/chr22/ASW.npz')
    # An earlier timestamp received in a later synchronization cannot switch a winner.
    create_gene(hosts['betty'], epoch=98, value=101)
    create_gene(hosts['betty'], chrom=21, epoch=105)
    result = c.collect(hosts, selected, gate)
    assert result['selected'] == 2 and result['newly_selected'] == 1
    assert c.sha(selected / 'genome/chr22/ASW.npz') == old_hash
    assert c.read_json(selected / 'winner_registry.json')['winners']['genome_chr22_ASW']['host'] == 'sesame'


@pytest.mark.parametrize('difference', ['source', 'vcf', 'settings'])
def test_reject_cross_host_campaign_or_input_mismatch(tmp_path, difference):
    hosts, gate = hosts_and_gate(tmp_path)
    create_gene(hosts['betty'])
    create_gene(hosts['sesame'], vcf_sha='different' if difference == 'vcf' else 'v' * 64,
                mu=2e-8 if difference == 'settings' else 1.25e-8)
    if difference == 'source':
        write_json(hosts['sesame'] / 'source/SOURCE_MANIFEST.json', {'git_commit': 'other'})
    with pytest.raises(ValueError, match='mismatch|differ'):
        c.collect(hosts, tmp_path / 'selected', gate)
    assert not list((tmp_path / 'selected').glob('genome/**/*.npz'))


def test_corrupt_or_failed_completed_tasks_are_never_selected(tmp_path):
    hosts, gate = hosts_and_gate(tmp_path)
    folder = create_gene(hosts['betty'])
    (folder / 'ASW.csv').write_text('corrupted')
    create_gene(hosts['sesame'])
    write_json(hosts['sesame'] / 'state/genome_chr22_ASW.failed.json', {'error': 'bad'})
    result = c.collect(hosts, tmp_path / 'selected', gate)
    assert result['selected'] == 0
    assert 'invalid' in (tmp_path / 'selected/inspection.tsv').read_text()


def test_requires_passed_gate_and_refuses_changed_winner_or_stale_partial(tmp_path):
    hosts, gate = hosts_and_gate(tmp_path)
    create_gene(hosts['betty'])
    write_json(gate, {'passed': False})
    with pytest.raises(ValueError, match='passed=true'):
        c.collect(hosts, tmp_path / 'selected', gate)
    write_json(gate, {'passed': True})
    with pytest.raises(ValueError, match='missing its source binding'):
        c.collect(hosts, tmp_path / 'selected', gate)
    write_json(gate, {'passed': True, 'source_manifest_sha256': c.sha(hosts['betty']/'source/SOURCE_MANIFEST.json')})
    c.collect(hosts, tmp_path / 'selected', gate)
    (tmp_path / 'selected/genome/chr22/ASW.csv').write_text('stale')
    with pytest.raises(ValueError, match='winner changed'):
        c.collect(hosts, tmp_path / 'selected', gate)
    stale = tmp_path / 'stale'
    stale.mkdir(); (stale / 'unexpected.npz').write_bytes(b'partial')
    with pytest.raises(ValueError, match='stale partial'):
        c.collect(hosts, stale, gate)


def test_candidate_aggregation_rejects_cross_population_settings(tmp_path):
    hosts, _ = hosts_and_gate(tmp_path)
    first = create_gene(hosts['betty'], pop='ASW')
    second = create_gene(hosts['betty'], pop='ACB', mu=2e-8)
    import shutil
    results = tmp_path / 'results'
    (results / 'chr22').mkdir(parents=True)
    for folder in (first, second):
        for path in folder.iterdir():
            shutil.copyfile(path, results / 'chr22' / path.name)
    from build_candidates import aggregate
    with pytest.raises(ValueError, match='Mixed scientific campaigns'):
        aggregate(results, [22], ['ASW', 'ACB'])


def test_production_source_is_separate_from_old_benchmark_snapshot(tmp_path):
    hosts, gate = hosts_and_gate(tmp_path)
    # The old source/ benchmark snapshots may differ. source-v2 is identical.
    for root in hosts.values():
        shutil.copytree(root/'source', root/'source-v2')
        (root/'source-v2/production.py').write_text('production')
        write_json(root / 'source-v2/SOURCE_MANIFEST.json', {'git_commit':'b'*40, 'files': {
            'production.py': c.sha(root/'source-v2/production.py'), 'infer.py': c.sha(root/'source-v2/infer.py')}})
    create_gene(hosts['betty'])
    prod_hash = c.sha(hosts['betty'] / 'source-v2/SOURCE_MANIFEST.json')
    marker = hosts['betty'] / 'state/genome_chr22_ASW.done.json'
    done = c.read_json(marker); done['source_manifest_sha256'] = prod_hash
    write_json(marker, done)
    write_json(hosts['sesame'] / 'source/SOURCE_MANIFEST.json', {'old_benchmark': 'other'})
    # A gate bound to the old source cannot authorize production collection.
    with pytest.raises(ValueError, match='different frozen source'):
        c.collect(hosts, tmp_path / 'selected', gate, 'source-v2')
    write_json(gate, {'passed': True, 'source_manifest_sha256': prod_hash})
    result = c.collect(hosts, tmp_path / 'selected', gate, 'source-v2')
    assert result['selected'] == 1
    assert c.read_json(tmp_path / 'selected/winner_registry.json')['source_directory'] == 'source-v2'


def single_host_local_gate(tmp_path):
    hosts, _ = hosts_and_gate(tmp_path)
    root = hosts['betty']
    shutil.copytree(root/'source', root/'source-v2')
    folder = create_gene(root)
    ready = c.read_json(folder/'ASW.metadata.json')['identity']['inputs']['cache']['record']
    write_json(root/'inputs/cache/parsed/chr22/READY.json', ready)
    preflight = root/'validation/preflight-v2'
    preflight.mkdir(parents=True)
    pairs = np.arange(40).reshape(20, 2)
    full = np.full((2000, 20), 100., dtype=np.float32)
    np.savez(preflight/'cross_host.npz', positions=np.arange(2000), pairs=pairs, full=full, blockwise=full)
    (preflight/'gene_comparison.csv').write_text('gene_name,n_sites,full,blockwise,abs_log_error\nGENE,2,100.,100.,0.\n')
    d = dict(passed=True, hostname='betty-gpu-node', gpu='0', samples=74, pairs=pairs.tolist(), retained_sites=170000,
             settings=dict(chr=22, pop='ASW', core_sites=65536, flank_sites=8192),
             calibration=dict(time_units='generations', physical_mu=1.25e-8, physical_rho=1e-8),
             full_metadata=dict(time_units='generations'), blockwise_metadata=dict(time_units='generations'),
             diagnostic_limits=dict(site_p99_abs_log=.01, site_max_abs_log=.1, gene_max_abs_log=.01),
             site_abs_log_error=dict(median=0., p95=0., p99=0., max=0.), gene_max_abs_log_error=0.)
    write_json(preflight/'validation.json', d)
    (root/'validation/gpu-tests.exit').write_text('0\n')
    gate = root/'validation/production-approved.json'
    write_json(gate, dict(passed=True, source_manifest_sha256=c.sha(root/'source-v2/SOURCE_MANIFEST.json'),
                         preflight_sha256=c.sha(preflight/'validation.json'),
                         chromosome22_READY_sha256=c.sha(root/'inputs/cache/parsed/chr22/READY.json'),
                         hostname='betty-gpu-node'))
    return {'betty': root}, gate


def test_single_host_local_gate_collects_without_claiming_cross_hardware_parity(tmp_path):
    hosts, gate = single_host_local_gate(tmp_path)
    selected = tmp_path/'selected'
    result = c.collect(hosts, selected, source_directory='source-v2', local_validation=gate)
    assert result['selected'] == 1
    assert result['cross_host_parity_validated'] is False
    registry = c.read_json(selected/'winner_registry.json')
    assert registry['validation_mode'] == 'single_host_local_gpu'
    assert 'cross_host_validation' not in registry
    assert registry['validation_evidence']['retained_sample_max_abs_log'] == 0
    assert 'No cross-host or cross-hardware parity claim' in registry['validation_evidence']['scope']
    assert len(registry['validation_evidence']['artifact_sha256']) == 6
    # An explicitly local campaign can be extended without switching its winner.
    create_gene(hosts['betty'], chrom=21)
    assert c.collect(hosts, selected, source_directory='source-v2', local_validation=gate)['selected'] == 2
    # A separate cross-host selection must use a new output registry.
    with pytest.raises(ValueError, match='another source campaign or validation gate'):
        c.collect(hosts, selected, gate, 'source-v2')


def test_retained_subset_percentile_can_exceed_passed_dense_percentile(tmp_path):
    hosts, gate = single_host_local_gate(tmp_path)
    root = hosts['betty']
    preflight = root/'validation/preflight-v2'
    sample = preflight/'cross_host.npz'
    with np.load(sample) as data:
        arrays = {k: data[k] for k in data.files}
    # 420 out of 40,000 sampled cells exceed .01. They can be less than 1%
    # of the full 170,000-by-20 chromosome, whose p99 therefore still passes.
    arrays['blockwise'][:21] *= np.exp(.02)
    np.savez(sample, **arrays)
    diagnostic = c.read_json(preflight/'validation.json')
    diagnostic['site_abs_log_error']['max'] = .03
    write_json(preflight/'validation.json', diagnostic)
    approval = c.read_json(gate)
    approval['preflight_sha256'] = c.sha(preflight/'validation.json')
    write_json(gate, approval)

    selected = tmp_path/'selected'
    result = c.collect(hosts, selected, source_directory='source-v2', local_validation=gate)
    assert result['selected'] == 1
    evidence = c.read_json(selected/'winner_registry.json')['validation_evidence']
    assert .01 < evidence['retained_sample_p99_abs_log'] < .03
    assert evidence['retained_sample_max_abs_log'] < .03


@pytest.mark.parametrize('problem', ['source_file', 'approval', 'preflight', 'ready', 'unit_exit',
                                     'array', 'gene', 'hostname', 'settings', 'metadata_source'])
def test_single_host_checks_real_gate_source_and_retained_evidence(tmp_path, problem):
    hosts, gate = single_host_local_gate(tmp_path)
    root = hosts['betty']
    if problem == 'source_file':
        (root/'source-v2/infer.py').write_text('changed after approval')
    elif problem == 'approval':
        d = c.read_json(gate); d['source_manifest_sha256'] = 'other'; write_json(gate, d)
    elif problem == 'preflight':
        d = c.read_json(root/'validation/preflight-v2/validation.json'); d['passed'] = False
        write_json(root/'validation/preflight-v2/validation.json', d)
    elif problem == 'ready':
        write_json(root/'inputs/cache/parsed/chr22/READY.json', {'different': True})
    elif problem == 'unit_exit':
        (root/'validation/gpu-tests.exit').write_text('1')
    elif problem == 'array':
        p = root/'validation/preflight-v2/cross_host.npz'
        with np.load(p) as data:
            arrays = {k: data[k] for k in data.files}
        arrays['blockwise'][0, 0] = 1000
        np.savez(p, **arrays)
    elif problem == 'gene':
        (root/'validation/preflight-v2/gene_comparison.csv').write_text('gene_name,n_sites,full,blockwise,abs_log_error\nGENE,2,100.,101.,0.\n')
    elif problem == 'hostname':
        d = c.read_json(gate); d['hostname'] = 'different'; write_json(gate, d)
    elif problem == 'settings':
        create_gene(root, mu=2e-8)
    else:
        path = root/'outputs/tasks/genome_chr22_ASW/attempt-001/results/chr22/ASW.metadata.json'
        meta = c.read_json(path); meta['identity']['source_sha256'] = {'infer.py': 'unapproved'}; write_json(path, meta)
        marker = root/'state/genome_chr22_ASW.done.json'
        done = c.read_json(marker)
        done['outputs']['results/chr22/ASW.metadata.json']['sha256'] = c.sha(path)
        done['outputs']['results/chr22/ASW.metadata.json']['bytes'] = path.stat().st_size
        write_json(marker, done)
    with pytest.raises(ValueError):
        c.collect(hosts, tmp_path/'selected', source_directory='source-v2', local_validation=gate)
    assert not list((tmp_path/'selected').glob('genome/**/*.npz'))


def test_local_mode_refuses_multiple_hosts_and_derived_runtime_approval(tmp_path):
    hosts, gate = single_host_local_gate(tmp_path)
    with pytest.raises(ValueError, match='exactly one host'):
        c.collect({**hosts, 'sesame': tmp_path/'sesame'}, tmp_path/'selected', source_directory='source-v2', local_validation=gate)
    with pytest.raises(ValueError, match='no derived/runtime-v3'):
        c.collect(hosts, tmp_path/'selected', source_directory='source-runtime-v3', local_validation=gate)
    with pytest.raises(ValueError, match='exactly one cross-host or local'):
        c.collect(hosts, tmp_path/'selected', gate, 'source-v2', local_validation=gate)


def test_single_host_preserves_artifact_validation_and_separate_source_scopes(tmp_path):
    hosts, gate = single_host_local_gate(tmp_path)
    root = hosts['betty']
    (root/'outputs/tasks/genome_chr22_ASW/attempt-001/results/chr22/ASW.csv').write_text('corrupted')
    selected = tmp_path/'selected'
    assert c.collect(hosts, selected, source_directory='source-v2', local_validation=gate)['selected'] == 0
    assert 'mismatched artifact' in (selected/'inspection.tsv').read_text()
    # No ASMC replacements are folded into the genome-only 572-task registry.
    write_json(root/'state/orthogonal-v3-asmc-GENE_ASW.done.json', {'source_manifest_sha256': 'runtime-v3'})
    create_gene(root)
    assert c.collect(hosts, selected, source_directory='source-v2', local_validation=gate)['selected'] == 1
    assert set(c.read_json(selected/'winner_registry.json')['winners']) == {'genome_chr22_ASW'}
