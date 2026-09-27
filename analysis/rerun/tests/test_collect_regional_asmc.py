"""Manifest-scoped supersession must never fall back to historical ASMC."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1]/'collect_regional_asmc.py'
spec = importlib.util.spec_from_file_location('regional_collector', FILE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fingerprint(root, path):
    return dict(path=str(Path('/remote/campaign')/path.relative_to(root)), size=path.stat().st_size, sha256=c.sha(path))


def task(gene, pop, action):
    command = ['{python}', '{source}/analysis/rerun/orthogonal.py', action,
               '--inputs-root', '{inputs}', '--output-root', '{root}/outputs/orthogonal',
               '--tools-root', '{tools}', '--chr', '1', '--pop', pop, '--gene', gene,
               '--pair-cap', '20', '--mu', '1.25e-08', '--rho', '1e-08']
    if action == 'asmc':
        command += ['--decoding-quantities', '{tools}/asmc_data/CEU_csfs50/decoding.decodingQuantities.gz']
    return dict(id=f'orthogonal-{action}-{gene}_{pop}', stage='orthogonal_'+action,
                command=command, kind='gpu' if action == 'regional' else 'cpu', requires=[],
                expected=['{root}/outputs/orthogonal/'+action+f'/{gene}_{pop}/COMPLETE.json'])


def finish_task(root, definition, scope, manifest_hash, source_hash, folder, inputs, details):
    remote = Path('/remote/campaign')
    values = dict(root=str(remote), inputs=str(remote/'inputs'), tools=str(remote/'inputs/tools'),
                  source=str(remote/scope), output=str(remote/'outputs/tasks'/definition['id']/'attempt-001'), python='/python')
    config = {'action': definition['command'][2], 'half_window_bp':500000}
    for key, value in c.flags(definition).items():
        value = value.format(**values)
        if key in ['--chr', '--pair-cap']:
            value = int(value)
        elif key in ['--mu', '--rho']:
            value = float(value)
        config[key[2:].replace('-', '_')] = value
    complete = dict(complete=True, identity={'configuration':config, 'inputs':inputs},
                    outputs={'results.npz':c.sha(folder/'results.npz')}, details=details)
    write(folder/'COMPLETE.json', complete)
    done = dict(task_id=definition['id'], stage=definition['stage'], returncode=0,
                completed_epoch=100., source_manifest_sha256=source_hash, task_manifest_sha256=manifest_hash,
                output=values['output'], command=[s.format(**values) for s in definition['command']],
                outputs={definition['expected'][0]:dict(sha256=c.sha(folder/'COMPLETE.json'), bytes=(folder/'COMPLETE.json').stat().st_size)})
    write(root/'state'/f"{definition['id']}.done.json", done)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/'host'
    targets = [(f'GENE{i}', 1, 'CEU', 'case') for i in range(18)]
    scopes = {}
    for scope in ('source-v2', 'source-runtime-v3'):
        path = root/scope/'analysis/rerun/orthogonal.py'
        path.parent.mkdir(parents=True)
        path.write_text('TARGETS = '+repr(targets)+'\n# '+scope+'\n')
        write(root/scope/'SOURCE_MANIFEST.json', {'git_commit':'a'*40, 'files':{'analysis/rerun/orthogonal.py':c.sha(path)}})
        scopes[scope] = c.sha(root/scope/'SOURCE_MANIFEST.json')
    old_tasks = [task(g, pop, action) for g, _, focal, _ in targets for pop in (focal, 'YRI') for action in ('regional', 'asmc')]
    old_path = root/'control/tasks-orthogonal.json'
    write(old_path, dict(source='source-v2', source_manifest_sha256=scopes['source-v2'], tasks=old_tasks))
    new_tasks = []
    for original in old_tasks:
        if original['stage'] != 'orthogonal_asmc':
            continue
        t = copy.deepcopy(original)
        t['id'] = t['id'].replace('orthogonal-asmc-', 'orthogonal-v3-asmc-')
        t['stage'] = 'orthogonal_asmc_fixed_seed'
        t['command'][t['command'].index('--output-root')+1] = '{root}/outputs/orthogonal-asmc-v3'
        t['command'] += ['--regional-root', '{root}/outputs/orthogonal']
        t['expected'] = [s.replace('/outputs/orthogonal/asmc/', '/outputs/orthogonal-asmc-v3/asmc/') for s in t['expected']]
        new_tasks.append(t)
    new_path = root/'control/tasks-asmc-regional-v3.json'
    write(new_path, dict(source='source-runtime-v3', source_manifest_sha256=scopes['source-runtime-v3'],
                         tasks=new_tasks, replaces={'original_manifest_sha256':c.sha(old_path), 'stage':'orthogonal_asmc'}))
    samples = root/'inputs/samples.txt'
    samples.parent.mkdir(parents=True)
    ids = [f'S{i}' for i in range(102)]
    samples.write_text('header\n'+''.join(f'x {s} x x x {"CEU" if i < 51 else "YRI"} x\n' for i, s in enumerate(ids)))
    parsed = root/'inputs/cache/parsed/chr1'
    parsed.mkdir(parents=True)
    positions = np.array([100, 200, 300], dtype=np.int64)
    np.save(parsed/'positions.npy', positions)
    np.save(parsed/'sample_ids.npy', ids)
    ready = dict(positions_base=1, positions_dtype='int64', members={name:dict(sha256=c.sha(parsed/name)) for name in ('positions.npy', 'sample_ids.npy')})
    write(parsed/'READY.json', ready)
    genes = root/'inputs/cache/genes/chr1_genes.tsv'
    genes.parent.mkdir(parents=True)
    genes.write_text('gene_name\tstart\tend\n'+''.join(f'GENE{i}\t{100+i}\t{200+i}\n' for i in range(18)))
    dq = root/'inputs/tools/asmc_data/CEU_csfs50/decoding.decodingQuantities.gz'
    dq.parent.mkdir(parents=True)
    dq.write_bytes(b'fictional DQ')
    native = dict(native_parameters=c.NATIVE, native_csfs_seed=1234, state_count=3, expected_times_generations=[1., 10., 100.])
    smoke = root/'validation/asmc-regional-v3-native'
    smoke.mkdir(parents=True)
    np.save(smoke/'means.npy', np.ones(2048, dtype=np.float32))
    write(root/'outputs/runtime/panel/manifest.json', {'fictional':'panel'})
    write(smoke/'validation.json', dict(passed=True, exact_repeatability=True, source_manifest_sha256=scopes['source-runtime-v3'],
                                      panel_manifest_sha256=c.sha(root/'outputs/runtime/panel/manifest.json'),
                                      means_sha256=c.sha(smoke/'means.npy'), n_sites=2048, metadata=native, dq_sha256=c.sha(dq)))
    for i, (gene, _, focal, _) in enumerate(targets):
        for pop in (focal, 'YRI'):
            suffix = f'{gene}_{pop}'
            haps = np.arange(102) if pop == 'CEU' else np.arange(102, 204)
            all_pairs = np.array([(a, b) for a in range(102) for b in range(a+1, 102)])
            pairs = all_pairs[np.sort(np.random.default_rng(42).choice(len(all_pairs), 20, replace=False))]
            rng = np.random.default_rng(123)
            subsets = [[a, b]+sorted(rng.choice([x for x in range(102) if x not in (a,b)], 98, replace=False).tolist()) for a, b in pairs]
            common_inputs = dict(samples=fingerprint(root, samples), genes=fingerprint(root, genes), cache_manifest=fingerprint(root, parsed/'READY.json'))
            reg = root/'outputs/orthogonal/regional'/suffix
            reg.mkdir(parents=True)
            np.savez(reg/'results.npz', gene=gene, chromosome=1, population=pop, gene_start=100+i, gene_end=200+i,
                     window_start=150+i-500000, window_end=150+i+500000, population_hap_indices=haps,
                     pairs=pairs, window_positions=positions, gamma_positions=positions,
                     gamma_mean=np.full((3, 20), 100.), cxt_log_tmrca=np.ones((20,4)))
            definition = next(t for t in old_tasks if t['id'] == 'orthogonal-regional-'+suffix)
            finish_task(root, definition, 'source-v2', c.sha(old_path), scopes['source-v2'], reg,
                        {**common_inputs, 'source':fingerprint(root, root/'source-v2/analysis/rerun/orthogonal.py')},
                        {'cache':{'record':ready}})
            # One historical product exists; the other35 were never produced.
            for historical in ([False, True] if suffix == 'GENE0_CEU' else [False]):
                scope = 'source-v2' if historical else 'source-runtime-v3'
                folder = root/('outputs/orthogonal/asmc' if historical else 'outputs/orthogonal-asmc-v3/asmc')/suffix
                folder.mkdir(parents=True)
                np.savez(folder/'results.npz', positions=positions, pairs=pairs, subsets=subsets,
                         mean=np.full((20, 3), 101. if historical else 100.))
                definition = next(t for t in (old_tasks if historical else new_tasks) if t['id'] == ('orthogonal-asmc-' if historical else 'orthogonal-v3-asmc-')+suffix)
                inp = {**common_inputs, 'source':fingerprint(root, root/scope/'analysis/rerun/orthogonal.py'),
                       'regional_marker':fingerprint(root, reg/'COMPLETE.json'), 'decoding_quantities':fingerprint(root, dq)}
                details = dict(cache={'record':ready}, regional_dependency=fingerprint(root, reg/'COMPLETE.json'),
                               decoding_quantities=fingerprint(root, dq), asmc_version='1.4.0', subset_seed=123,
                               diploid_samples=50, csfs_haploid_sample_count=50, no_pair_failures_ignored=True,
                               time_units='generations from calibrated decoding quantities', **({} if historical else native))
                finish_task(root, definition, scope, c.sha(old_path if historical else new_path), scopes[scope], folder, inp, details)
    return root, old_path, new_path


def rehash_product(root, task_id, folder):
    marker = folder/'COMPLETE.json'
    complete = c.read(marker)
    complete['outputs']['results.npz'] = c.sha(folder/'results.npz')
    write(marker, complete)
    done_path = root/'state'/f'{task_id}.done.json'
    done = c.read(done_path)
    for entry in done['outputs'].values():
        entry.update(sha256=c.sha(marker), bytes=marker.stat().st_size)
    write(done_path, done)


def test_complete36_replacement_registry_preserves_originals_and_absence(campaign, tmp_path):
    root, old, new = campaign
    original = root/'outputs/orthogonal/asmc/GENE0_CEU/results.npz'
    old_hash = c.sha(original)
    result = c.collect(root, old, new, tmp_path/'selected')
    assert result['complete'], result['errors']
    assert len(result['replacements']) == 36
    assert sum(r['original']['historical_product_status'] == 'not_produced' for r in result['replacements'].values()) == 35
    assert c.sha(original) == old_hash
    assert np.load(original)['mean'][0,0] == 101.
    assert np.load(tmp_path/'selected/asmc/GENE0_CEU/results.npz')['mean'][0,0] == 100.
    assert result['manifests']['original']['source'] == 'source-v2'
    assert result['manifests']['replacement']['source'] == 'source-runtime-v3'
    assert all(len(r['selected_artifacts']) == 2 for r in result['replacements'].values())
    assert c.read(tmp_path/'selected/validation.json')['passed']
    with pytest.raises(FileExistsError):
        c.collect(root, old, new, tmp_path/'selected')


@pytest.mark.parametrize('problem', ['missing_new', 'failed_new', 'checksum', 'seed', 'subsets', 'nonfinite',
                                     'old_partial', 'old_pairs', 'source', 'manifest', 'count', 'gate', 'dependency', 'panel'])
def test_incomplete_or_mismatched_products_never_fall_back_to_old(campaign, tmp_path, problem):
    root, old, new = campaign
    tid = 'orthogonal-v3-asmc-GENE0_CEU'
    folder = root/'outputs/orthogonal-asmc-v3/asmc/GENE0_CEU'
    if problem == 'missing_new':
        (root/'state'/f'{tid}.done.json').unlink()
    elif problem == 'failed_new':
        write(root/'state'/f'{tid}.failed.json', {'error':'failed'})
    elif problem == 'checksum':
        (folder/'results.npz').write_bytes(b'corrupted')
    elif problem == 'seed':
        marker = c.read(folder/'COMPLETE.json'); marker['details']['native_csfs_seed'] = 0
        write(folder/'COMPLETE.json', marker); rehash_product(root, tid, folder)
    elif problem in ('subsets', 'nonfinite', 'old_pairs'):
        if problem == 'old_pairs':
            tid, folder = 'orthogonal-asmc-GENE0_CEU', root/'outputs/orthogonal/asmc/GENE0_CEU'
        with np.load(folder/'results.npz') as z:
            arrays = {k:z[k] for k in z.files}
        if problem == 'subsets':
            arrays['subsets'][0, 5] = arrays['subsets'][0, 0]
        elif problem == 'nonfinite':
            arrays['mean'][-1, -1] = np.nan
        else:
            arrays['pairs'][0, :] = arrays['pairs'][1, :]
        np.savez(folder/'results.npz', **arrays); rehash_product(root, tid, folder)
    elif problem == 'old_partial':
        (root/'outputs/orthogonal/asmc/GENE0_YRI').mkdir(parents=True)
    elif problem == 'source':
        (root/'source-runtime-v3/analysis/rerun/orthogonal.py').write_text('changed')
    elif problem in ('manifest', 'count'):
        manifest = c.read(new)
        if problem == 'count':manifest['tasks'].pop()
        else:manifest['tasks'][0]['command'][manifest['tasks'][0]['command'].index('--chr')+1] = '2'
        write(new, manifest)
    elif problem == 'gate':
        (root/'validation/asmc-regional-v3-native/means.npy').write_bytes(b'changed')
    elif problem == 'dependency':
        (root/'outputs/orthogonal/regional/GENE0_CEU/results.npz').write_bytes(b'changed')
    else:
        np.save(root/'inputs/cache/parsed/chr1/sample_ids.npy', ['wrong'])
    result = c.collect(root, old, new, tmp_path/'selected')
    assert not result['complete'] and result['errors']
    assert not list((tmp_path/'selected').glob('asmc/**/*.npz'))
    assert not c.read(tmp_path/'selected/validation.json')['passed']


def test_cli_failure_retains_report_and_exits_nonzero(campaign, tmp_path):
    root, old, new = campaign
    (root/'state/orthogonal-v3-asmc-GENE17_YRI.done.json').unlink()
    output = tmp_path/'selected'
    run = subprocess.run([sys.executable, str(FILE), '--root', str(root), '--original-manifest', str(old),
                          '--replacement-manifest', str(new), '--output-dir', str(output)], capture_output=True, text=True)
    assert run.returncode == 1
    assert not c.read(output/'validation.json')['passed']
    assert len(c.read(output/'replacement_registry.json')['replacements']) == 35
