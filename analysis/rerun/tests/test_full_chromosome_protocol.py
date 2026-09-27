"""Whole-chromosome results cannot inherit a blockwise task or approval."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import full_chromosome_protocol as p
spec = importlib.util.spec_from_file_location('full_collect', HERE/'collect.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True)+'\n')


@pytest.fixture
def campaign(tmp_path):
    hosts = {name: tmp_path/name for name in ('betty', 'sesame')}
    for root in hosts.values():
        source = root/'source-v2'
        sources = {}
        for name in ['analysis/genome_wide/infer_chromosome.py', 'analysis/genome_wide/rerun_support.py']:
            file = source/name; file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('# fictional frozen implementation\n'); sources[name] = c.sha(file)
        write(source/'SOURCE_MANIFEST.json', dict(git_commit='a'*40, files=sources))
    source = hosts['betty']/'source-v2/SOURCE_MANIFEST.json'
    old = tmp_path/'old.json'
    write(old, dict(source='source-v2', source_manifest_sha256=c.sha(source), tasks=[
        dict(id=f'genome_chr{chrom}_{pop}', resources=dict(cpus=4, memory_gb=96, walltime_hours=48))
        for chrom in range(1, 23) for pop in p.POPS]))
    original_bytes = old.read_bytes()
    manifest = tmp_path/'tasks-full.json'; write(manifest, p.generate(old, source))
    assert old.read_bytes() == original_bytes
    for root in hosts.values():
        write(root/p.MANIFEST_PATH, c.read_json(manifest))
    gate = tmp_path/'full-cross-host.json'
    write(gate, dict(passed=True, protocol=p.PROTOCOL, source_manifest_sha256=c.sha(source),
                     task_manifest_sha256=c.sha(manifest), scientific_settings=p.SETTINGS,
                     cross_host_parity_validated=True, validated_hosts=['betty', 'sesame']))
    return hosts, manifest, gate


def complete(root, manifest, chrom=22, pop='ASW'):
    task = p.validate_manifest(c.read_json(manifest))[chrom, pop]
    tid = task['id']; remote = Path('/remote/campaign')
    out = root/'outputs/tasks'/tid/'attempt-001'; folder = out/f'results/chr{chrom}'
    folder.mkdir(parents=True)
    np.savez(folder/f'{pop}.npz', gene_id=['g'], gene_name=['GENE'], count=[6],
             n_pairs_total=6, n_sites_per_gene=[2], log_sum=[6*np.log(100.)])
    (folder/f'{pop}.csv').write_text('gene_name,geom_mean_tmrca\nGENE,100\n')
    values = dict(root=str(remote), source=str(remote/'source-v2'), inputs=str(remote/'inputs'),
                  tools=str(remote/'inputs/tools'), output=str(remote/'outputs/tasks'/tid/'attempt-001'),
                  python='/env/bin/python')
    config = dict(chr=chrom, cache_dir=str(remote/'inputs/cache'), samples=str(remote/'inputs/samples.txt'),
                  output_dir=values['output']+'/results', core_block_sites=6000000, flank_sites=0,
                  pair_chunk=512, mu=1.25e-8, rho=1e-8, sequence_length=None,
                  lead_variants=str(remote/'inputs/akbari_lead_variants_grch38.tsv'), lead_half_bp=25000)
    cache = dict(schema=2, dtype='uint8', positions_dtype='int64', positions_base=1,
                 coordinate_convention='VCF POS one-based', source_sha256='v'*64, shape=[4, 2],
                 filters=dict(chromosome=str(chrom), rule='strict'),
                 members={n:dict(sha256=n*3) for n in ('G.npy', 'positions.npy', 'sample_ids.npy')})
    metadata = dict(complete=True, calibration=dict(time_units='generations', physical_mu=1.25e-8,
                    physical_rho=1e-8, calibration='heterozygosity', denominator_convention='last position + 1'),
        identity=dict(chromosome=chrom, population=pop, configuration=config,
                      source_sha256=c.read_json(root/'source-v2/SOURCE_MANIFEST.json')['files'],
                      inputs=dict(cache=dict(format='extracted_npy', record=cache), samples=dict(sha256='s'*64),
                                  genes=dict(sha256=f'genes{chrom}'), lead_variants=dict(sha256='l'*64))),
        output_sha256={ext:c.sha(folder/f'{pop}{ext}') for ext in ('.npz', '.csv')})
    write(folder/f'{pop}.metadata.json', metadata)
    done = dict(task_id=tid, stage=task['stage'], source_manifest_sha256=c.sha(root/'source-v2/SOURCE_MANIFEST.json'),
                task_manifest_sha256=c.sha(manifest), command=[s.format(**values) for s in task['command']],
                output=values['output'], returncode=0, completed_epoch=100., outputs={
                    n:dict(sha256=c.sha(out/n), bytes=(out/n).stat().st_size) for n in task['expected']})
    write(root/'state'/f'{tid}.done.json', done)
    return folder, root/'state'/f'{tid}.done.json'


def collect(campaign, output):
    hosts, manifest, gate = campaign
    return c.collect(hosts, output, gate, 'source-v2', full_task_manifest=manifest)


def test_exact572_mapping_new_ids_and_separate_incremental_collection(campaign, tmp_path):
    hosts, manifest, gate = campaign
    m = c.read_json(manifest)
    mapping = p.validate_manifest(m)
    assert len(mapping) == 572
    assert all(t['id'].startswith('genome-full_chr') for t in mapping.values())
    assert all(p.GATE in t['requires'] for t in mapping.values())
    assert m['source'] == 'source-v2'
    complete(hosts['sesame'], manifest)
    result = collect(campaign, tmp_path/'selected')
    assert result['selected'] == 1 and not result['complete']
    assert result['validation_mode'] == 'full_chromosome_cross_host'
    registry = c.read_json(tmp_path/'selected/winner_registry.json')
    assert set(registry['winners']) == {'genome-full_chr22_ASW'}
    assert registry['task_manifest_sha256'] == c.sha(manifest)
    assert registry['winners']['genome-full_chr22_ASW']['task_manifest_sha256'] == c.sha(manifest)
    complete(hosts['betty'], manifest, chrom=21)
    assert collect(campaign, tmp_path/'selected')['newly_selected'] == 1
    # Forgetting the mapping must not reinterpret a full gate as blockwise approval.
    with pytest.raises(ValueError, match='explicit full task manifest mapping'):
        c.collect(hosts, tmp_path/'wrong-new-output', gate, 'source-v2')
    # The original blockwise collector cannot reuse this winner registry.
    old_gate = tmp_path/'old-gate.json'
    write(old_gate, dict(passed=True, source_manifest_sha256=c.read_json(gate)['source_manifest_sha256']))
    with pytest.raises(ValueError, match='another source campaign'):
        c.collect(hosts, tmp_path/'selected', old_gate, 'source-v2')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'old_id', 'command', 'outputs', 'gate', 'source'])
def test_manifest_mapping_is_strict(campaign, change):
    _, manifest, _ = campaign
    m = c.read_json(manifest)
    if change == 'missing': m['tasks'].pop()
    elif change == 'duplicate': m['tasks'][-1] = copy.deepcopy(m['tasks'][0])
    elif change == 'old_id': m['tasks'][0]['id'] = m['tasks'][0]['id'].replace('genome-full_', 'genome_')
    elif change == 'command': m['tasks'][0]['command'][m['tasks'][0]['command'].index('--flank-sites')+1] = '8192'
    elif change == 'outputs': m['tasks'][0]['expected'][0] = 'old/results.npz'
    elif change == 'gate': m['validation_gate'] = '{root}/validation/production-approved.json'
    else: m['source'] = 'source-runtime-v3'
    with pytest.raises(ValueError): p.validate_manifest(m)


@pytest.mark.parametrize('change', ['gate_protocol', 'gate_manifest', 'gate_settings', 'gate_parity',
                                     'gate_hosts', 'host_manifest', 'receipt_manifest', 'receipt_command',
                                     'receipt_stage', 'receipt_outputs', 'metadata_settings', 'metadata_path',
                                     'calibration', 'raw_size'])
def test_full_protocol_rejects_relabelled_or_changed_evidence(campaign, tmp_path, change):
    hosts, manifest, gate = campaign
    root = hosts['sesame']; folder, marker = complete(root, manifest)
    if change.startswith('gate_'):
        data = c.read_json(gate)
        key, value = {'gate_protocol':('protocol','old_blockwise'),
                      'gate_manifest':('task_manifest_sha256','a'*64),
                      'gate_settings':('scientific_settings',{**p.SETTINGS,'flank_sites':8192}),
                      'gate_parity':('cross_host_parity_validated',False),
                      'gate_hosts':('validated_hosts',['betty'])}[change]
        data[key] = value; write(gate, data)
    elif change == 'host_manifest':
        (root/p.MANIFEST_PATH).write_text('{}')
    elif change.startswith('receipt_'):
        data = c.read_json(marker)
        if change == 'receipt_manifest':data['task_manifest_sha256'] = 'old-manifest'
        elif change == 'receipt_stage':data['stage'] = 'genome'
        elif change == 'receipt_outputs':data['outputs'].pop(next(iter(data['outputs'])))
        else:data['command'][data['command'].index('--core-block-sites')+1] = '65536'
        write(marker, data)
    else:
        meta = folder/'ASW.metadata.json';data = c.read_json(meta)
        if change == 'metadata_settings':data['identity']['configuration']['flank_sites'] = 8192
        elif change == 'metadata_path':data['identity']['configuration']['output_dir'] = '/old/results'
        elif change == 'calibration':data['calibration']['physical_mu'] = 2e-8
        else:data['identity']['inputs']['cache']['record']['shape'][1] = 6000001
        write(meta, data)
        receipt = c.read_json(marker);item = receipt['outputs']['results/chr22/ASW.metadata.json']
        item.update(sha256=c.sha(meta), bytes=meta.stat().st_size);write(marker, receipt)
    with pytest.raises(ValueError):collect(campaign, tmp_path/'selected')
    assert not list((tmp_path/'selected').glob('genome/**/*.npz'))


def test_full_mode_preserves_hash_checks_and_does_not_accept_local_gate(campaign, tmp_path):
    hosts, manifest, gate = campaign
    folder, _ = complete(hosts['sesame'], manifest)
    (folder/'ASW.npz').write_bytes(b'corrupted')
    assert collect(campaign, tmp_path/'selected')['selected'] == 0
    with pytest.raises(ValueError, match='explicit cross-host gate'):
        c.collect(hosts, tmp_path/'other', source_directory='source-v2',
                  local_validation=gate, full_task_manifest=manifest)
