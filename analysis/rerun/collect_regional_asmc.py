"""Collect exactly 36 seeded ASMC replacements from one synchronized host.

Original products remain untouched. No old result is ever used as a fallback.
This is a manifest-scoped regional replacement registry, not a genome collector
or a cross-host numerical-parity claim.
"""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


NATIVE = dict(jobs=1, jobInd=1, decodingSequence=True, usingCSFS=True, compress=False,
              skipCSFSdistance=0., noBatches=False, doPerPairPosteriorMean=True,
              doPosteriorSums=False, doMajorMinorPosteriorSums=False, doPerPairMAP=False,
              useKnownSeed=True, batchSize=64)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = Path(path).with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def flags(task):
    command = task['command']
    require(len(command) >= 3 and len(command[3:]) % 2 == 0, 'Invalid orthogonal task command')
    result = dict(zip(command[3::2], command[4::2]))
    require(len(result)*2 == len(command[3:]), 'Duplicate command flags')
    return result


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.artifacts = {}

    def path(self, relative):
        path = (self.root/relative).resolve()
        require(path.is_relative_to(self.root), f'Path escapes synchronized root: {relative}')
        return path

    def verify(self, path, digest=None, size=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root) and path.is_file(), f'Missing/outside artifact: {path}')
        actual = sha(path)
        require(digest is None or actual == digest, f'Artifact hash mismatch: {path}')
        require(size is None or path.stat().st_size == size, f'Artifact size mismatch: {path}')
        self.artifacts[str(path.relative_to(self.root))] = dict(sha256=actual, bytes=path.stat().st_size)
        return actual

    def source(self, name, expected):
        source = self.path(name)
        self.verify(source/'SOURCE_MANIFEST.json', expected)
        frozen = read(source/'SOURCE_MANIFEST.json')
        require(re.fullmatch('[0-9a-fA-F]{40}', frozen.get('git_commit', '')) and frozen.get('files'), 'Invalid frozen source')
        hashes = {}
        for relative, entry in frozen['files'].items():
            path = (source/relative).resolve()
            require(path.is_relative_to(source), 'Source manifest path escapes snapshot')
            digest = entry['sha256'] if isinstance(entry, dict) else entry
            hashes[relative] = self.verify(path, digest)
        return hashes

    def task(self, task, manifest_hash, source_hash, source_name, optional=False):
        tid = task['id']
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', tid), 'Unsafe task ID')
        local_values = dict(root=str(self.root), source=str(self.root/source_name),
                            inputs=str(self.root/'inputs'), tools=str(self.root/'inputs/tools'),
                            output=str(self.root/'outputs/tasks'/tid/'attempt-001'), python='unused')
        require(len(task['expected']) == 1, f'Expected one scientific marker: {tid}')
        marker = self.path(task['expected'][0].format(**local_values))
        done_path = self.path(f'state/{tid}.done.json')
        if optional and not done_path.exists() and not marker.parent.exists():
            return {'historical_product_status': 'not_produced', 'task_id': tid,
                    'task_manifest_sha256': manifest_hash, 'source_manifest_sha256': source_hash,
                    'absence_scope': 'No successful receipt or product directory in the supplied synchronized snapshot; no assertion about unsynchronized hosts.'}, None
        require(not self.path(f'state/{tid}.failed.json').exists(), f'Task has a failed marker: {tid}')
        self.verify(done_path)
        done = read(done_path)
        require(done.get('task_id') == tid and done.get('stage') == task['stage']
                and done.get('returncode') == 0 and not done.get('error'), f'Invalid successful receipt: {tid}')
        require(done.get('task_manifest_sha256') == manifest_hash and done.get('source_manifest_sha256') == source_hash,
                f'Task/source manifest identity mismatch: {tid}')
        require(isinstance(done.get('completed_epoch'), (int, float)) and math.isfinite(done['completed_epoch'])
                and done['completed_epoch'] > 0, f'Invalid completion time: {tid}')
        recorded = Path(done['output'])
        suffix = Path('outputs/tasks')/tid/'attempt-001'
        require(recorded.is_absolute() and recorded.parts[-4:] == suffix.parts, f'Invalid attempt path: {tid}')
        remote = recorded.parents[3]
        values = dict(root=str(remote), source=str(remote/source_name), inputs=str(remote/'inputs'),
                      tools=str(remote/'inputs/tools'), output=str(recorded), python=done['command'][0])
        require(done['command'] == [token.format(**values) for token in task['command']], f'Executed command differs from task: {tid}')
        require(set(done['outputs']) == set(task['expected']), f'Worker expected-artifact inventory changed: {tid}')
        detail = done['outputs'][task['expected'][0]]
        self.verify(marker, detail['sha256'], detail['bytes'])
        complete = read(marker)
        require(complete.get('complete') is True and set(complete['outputs']) == {'results.npz'}, f'Invalid scientific completion marker: {tid}')
        self.verify(marker.parent/'results.npz', complete['outputs']['results.npz'])
        # Every declared fingerprint must resolve to the same synchronized input.
        for fingerprint in complete['identity']['inputs'].values():
            require(isinstance(fingerprint, dict) and 'path' in fingerprint and 'sha256' in fingerprint,
                    f'Unexpected input fingerprint schema: {tid}')
            path = Path(fingerprint['path'])
            require(path.is_absolute() and path.is_relative_to(remote), f'Input outside recorded campaign: {tid}')
            self.verify(self.path(path.relative_to(remote)), fingerprint['sha256'], fingerprint.get('size'))
        config = complete['identity']['configuration']
        command_flags = flags(task)
        for option, expected in command_flags.items():
            key = option.removeprefix('--').replace('-', '_')
            expected = expected.format(**values)
            actual = config.get(key)
            if isinstance(actual, (int, float)):
                require(float(expected) == actual, f'Configuration differs for {tid}: {option}')
            else:
                require(str(actual) == expected, f'Configuration differs for {tid}: {option}')
        require(config.get('action') == task['command'][2], f'Wrong scientific action: {tid}')
        require(config.get('half_window_bp') == 500000, f'Unexpected regional window definition: {tid}')
        return dict(historical_product_status='produced', task_id=tid, source_manifest_sha256=source_hash,
                    task_manifest_sha256=manifest_hash, done_sha256=sha(done_path),
                    complete_sha256=sha(marker), results_sha256=complete['outputs']['results.npz'],
                    completed_epoch=done['completed_epoch'], marker=str(marker.relative_to(self.root)),
                    results=str((marker.parent/'results.npz').relative_to(self.root))), complete


def check_native(details, expected_grid=None):
    require(details.get('native_parameters') == NATIVE and details.get('native_csfs_seed') == 1234,
            'Corrected ASMC must use exact full-CSFS native settings and seed1234')
    grid = np.asarray(details['expected_times_generations'], dtype=float)
    require(grid.ndim == 1 and len(grid) == details['state_count'] and len(grid) > 1
            and np.isfinite(grid).all() and np.all(grid > 0) and np.all(np.diff(grid) > 0), 'Invalid ASMC time-state grid')
    require(expected_grid is None or np.array_equal(grid, expected_grid), 'ASMC state grid differs from actual native gate')
    return grid


def check_panel(audit, target, regional, regional_record, corrected, corrected_record, grid, dq_hash):
    gene, chrom, pop = target
    inputs = corrected['identity']['inputs']
    for key in ('samples', 'genes', 'cache_manifest'):
        require(inputs[key]['sha256'] == regional['identity']['inputs'][key]['sha256'], f'Regional input panel changed: {gene}/{pop}/{key}')
    for key, path in [('samples', 'inputs/samples.txt'),
                      ('genes', f'inputs/cache/genes/chr{chrom}_genes.tsv'),
                      ('cache_manifest', f'inputs/cache/parsed/chr{chrom}/READY.json')]:
        audit.verify(audit.path(path), inputs[key]['sha256'])
    require(inputs['regional_marker']['sha256'] == regional_record['complete_sha256'], 'Regional dependency marker mismatch')
    details = corrected['details']
    require(details.get('regional_dependency', {}).get('sha256') == regional_record['complete_sha256'], 'Regional dependency details mismatch')
    require(details.get('asmc_version') == '1.4.0' and details.get('subset_seed') == 123
            and details.get('diploid_samples') == 50 and details.get('csfs_haploid_sample_count') == 50
            and details.get('no_pair_failures_ignored') is True
            and details.get('time_units') == 'generations from calibrated decoding quantities', 'Incorrect regional ASMC model/subset convention')
    require(details['decoding_quantities']['sha256'] == inputs['decoding_quantities']['sha256'] == dq_hash, 'Decoding quantities differ from actual native gate')
    check_native(details, grid)
    ready_path = audit.path(f'inputs/cache/parsed/chr{chrom}/READY.json')
    ready = read(ready_path)
    require(details['cache']['record'] == regional['details']['cache']['record'] == ready, 'Input READY identities differ')
    require(ready.get('positions_base') == 1 and ready.get('positions_dtype') == 'int64', 'Expected exact one-based VCF coordinates')
    folder = ready_path.parent
    for name in ('sample_ids.npy', 'positions.npy'):
        audit.verify(folder/name, ready['members'][name]['sha256'])
    sample_ids = np.load(folder/'sample_ids.npy', allow_pickle=False)
    require(sample_ids.ndim == 1 and len(set(map(str, sample_ids))) == len(sample_ids), 'Invalid/duplicate cache sample identities')
    membership = {}
    with audit.path('inputs/samples.txt').open() as stream:
        next(stream)
        for line in stream:
            fields = line.split()
            if len(fields) >= 7:
                require(fields[1] not in membership, 'Duplicate sample population assignment')
                membership[fields[1]] = fields[5]
    require(all(str(s) in membership for s in sample_ids), 'Missing cohort population assignments')
    haps = np.asarray([j for i, s in enumerate(sample_ids) if membership[str(s)] == pop for j in (2*i, 2*i+1)])
    require(len(haps) >= 100, 'Regional panel has fewer than100 observed haplotypes')
    with audit.path(f'inputs/cache/genes/chr{chrom}_genes.tsv').open() as stream:
        genes = [r for r in csv.DictReader(stream, delimiter='\t') if r['gene_name'] == gene]
    require(len(genes) == 1, f'Expected exactly one locus annotation: {gene}')
    gs, ge = int(genes[0]['start']), int(genes[0]['end'])
    mid = (gs+ge)//2
    lo, hi = mid-500000, mid+500000
    positions = np.load(folder/'positions.npy', mmap_mode='r', allow_pickle=False)
    require(positions.ndim == 1 and positions.dtype == np.int64, 'Cache positions are not exact int64')
    expected_positions = positions[np.searchsorted(positions, lo):np.searchsorted(positions, hi, side='right')]
    require(len(expected_positions) > 0 and np.all(np.diff(expected_positions) > 0), 'Empty/invalid locus position grid')
    all_pairs = np.asarray([(i, j) for i in range(len(haps)) for j in range(i+1, len(haps))])
    cap = corrected['identity']['configuration']['pair_cap']
    require(cap == regional['identity']['configuration']['pair_cap'] == 20, 'Unexpected focal pair cap')
    pairs = all_pairs[np.sort(np.random.default_rng(42).choice(len(all_pairs), min(cap, len(all_pairs)), replace=False))]
    with np.load(audit.path(regional_record['results']), allow_pickle=False) as z:
        for name, expected in [('gene', gene), ('chromosome', chrom), ('population', pop),
                               ('gene_start', gs), ('gene_end', ge), ('window_start', lo), ('window_end', hi)]:
            require(z[name].item() == expected, f'Regional locus identity mismatch: {gene}/{name}')
        require(np.array_equal(z['population_hap_indices'], haps) and np.array_equal(z['pairs'], pairs), 'Regional population/pairs mismatch')
        require(np.array_equal(z['window_positions'], expected_positions), 'Regional locus grid differs from immutable input')
        gp, gm, cx = z['gamma_positions'], z['gamma_mean'], z['cxt_log_tmrca']
        require(gp.ndim == 1 and len(gp) > 0 and np.all(np.diff(gp) > 0)
                and np.isin(gp, expected_positions).all() and gm.shape == (len(gp), len(pairs))
                and np.isfinite(gm).all() and np.all(gm > 0)
                and cx.shape[0] == len(pairs) and cx.size > 0 and np.isfinite(cx).all(), 'Invalid original Gamma-SMC/CXT dependency')
    rng, subsets = np.random.default_rng(123), []
    for first, second in pairs:
        others = [i for i in range(len(haps)) if i not in (first, second)]
        subsets.append([first, second]+sorted(rng.choice(others, 98, replace=False).tolist()))
    with np.load(audit.path(corrected_record['results']), allow_pickle=False) as z:
        require(np.array_equal(z['positions'], expected_positions) and np.array_equal(z['pairs'], pairs)
                and np.array_equal(z['subsets'], subsets), 'Corrected ASMC locus/pairs/seed123 observed subsets mismatch')
        means = z['mean']
        require(means.shape == (len(pairs), len(expected_positions)) and np.isfinite(means).all() and np.all(means > 0),
                'Corrected ASMC posterior means are incomplete/nonpositive/nonfinite')
    return dict(gene=gene, chromosome=chrom, population=pop, gene_start=gs, gene_end=ge,
                window_start=lo, window_end=hi, n_sites=len(expected_positions), n_pairs=len(pairs),
                population_haplotypes=len(haps), subset_seed=123, native_csfs_seed=1234)


def collect(root, original_manifest, replacement_manifest, output):
    audit, output = Audit(root), Path(output).resolve()
    require(output != audit.root and not output.is_relative_to(audit.root) and not audit.root.is_relative_to(output), 'Output must be separate from synchronized host root')
    output.mkdir(parents=True, exist_ok=False)
    registry = dict(schema=1, complete=False, expected_replacements=36, replacements={}, errors=[],
                    scope='One synchronized host; explicit regional ASMC supersession only. No cross-host parity claim.',
                    input_root=str(audit.root), collector_sha256=sha(__file__))
    try:
        old_path, new_path = Path(original_manifest).resolve(), Path(replacement_manifest).resolve()
        old_hash, new_hash = audit.verify(old_path), audit.verify(new_path)
        old, new = read(old_path), read(new_path)
        require(old['source'] == 'source-v2' and new['source'] == 'source-runtime-v3', 'Unexpected original/replacement source scope')
        require(new['replaces']['original_manifest_sha256'] == old_hash and new['replaces']['stage'] == 'orthogonal_asmc', 'Incorrect replacement parent manifest')
        sources = {m['source']: audit.source(m['source'], m['source_manifest_sha256']) for m in (old, new)}
        registry['manifests'] = {label: dict(path=str(path.relative_to(audit.root)), sha256=digest,
                                  source=m['source'], source_manifest_sha256=m['source_manifest_sha256'])
                                  for label, m, path, digest in [('original', old, old_path, old_hash), ('replacement', new, new_path, new_hash)]}
        old_tasks = {t['id']:t for t in old['tasks']}
        require(len(old_tasks) == len(old['tasks']), 'Duplicate original task IDs')
        original = [t for t in old['tasks'] if t['stage'] == 'orthogonal_asmc']
        replacements = {t['id']:t for t in new['tasks']}
        require(len(original) == len(new['tasks']) == len(replacements) == 36, 'Require exactly36 unique regional ASMC replacements')
        tree = ast.parse(audit.path('source-v2/analysis/rerun/orthogonal.py').read_text())
        targets = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TARGETS' for t in n.targets)]
        require(len(targets) == 1, 'Cannot resolve frozen manuscript target definitions')
        expected_targets = {(g, int(ch), p) for g, ch, pop, group in targets[0] for p in (pop, 'YRI')}
        observed = {(flags(t)['--gene'], int(flags(t)['--chr']), flags(t)['--pop']) for t in original}
        require(observed == expected_targets and len(observed) == 36, 'Manifest differs from the frozen18 manuscript targets and their YRI comparisons')
        gate_path = audit.path('validation/asmc-regional-v3-native/validation.json')
        audit.verify(gate_path)
        gate = read(gate_path)
        require(gate.get('passed') is True and gate.get('exact_repeatability') is True
                and gate['source_manifest_sha256'] == new['source_manifest_sha256'], 'Actual regional-native gate is absent/failed/wrong-source')
        grid = check_native(gate['metadata'])
        audit.verify(audit.path('outputs/runtime/panel/manifest.json'), gate['panel_manifest_sha256'])
        means_path = gate_path.parent/'means.npy'
        audit.verify(means_path, gate['means_sha256'])
        means = np.load(means_path, allow_pickle=False)
        require(means.shape == (gate['n_sites'],) and gate['n_sites'] == 2048 and np.isfinite(means).all() and np.all(means > 0), 'Invalid actual native smoke means')
        registry['regional_native_gate_sha256'] = sha(gate_path)
        for task in original:
            f = flags(task)
            target = (f['--gene'], int(f['--chr']), f['--pop'])
            suffix = f'{target[0]}_{target[2]}'
            require(task['id'] == 'orthogonal-asmc-'+suffix, 'Old task ID does not match its target')
            expected = copy.deepcopy(task)
            expected['id'] = 'orthogonal-v3-asmc-'+suffix
            expected['stage'] = 'orthogonal_asmc_fixed_seed'
            expected['command'][expected['command'].index('--output-root')+1] = '{root}/outputs/orthogonal-asmc-v3'
            expected['command'] += ['--regional-root', '{root}/outputs/orthogonal']
            expected['expected'] = [s.replace('/outputs/orthogonal/asmc/', '/outputs/orthogonal-asmc-v3/asmc/') for s in task['expected']]
            replacement = replacements.get(expected['id'])
            require(replacement == expected, f'Replacement changed its declared scientific task: {suffix}')
            regional_task = old_tasks['orthogonal-regional-'+suffix]
            require(regional_task['stage'] == 'orthogonal_regional', 'Wrong original regional dependency stage')
            reg_record, regional = audit.task(regional_task, old_hash, old['source_manifest_sha256'], old['source'])
            history, historical = audit.task(task, old_hash, old['source_manifest_sha256'], old['source'], optional=True)
            fixed_record, corrected = audit.task(replacement, new_hash, new['source_manifest_sha256'], new['source'])
            for marker, scope in [(regional, old['source']), (corrected, new['source'])]+([(historical, old['source'])] if historical else []):
                require(marker['identity']['inputs']['source']['sha256'] == sources[scope]['analysis/rerun/orthogonal.py'], 'Scientific marker source is not its assigned frozen source')
            checked = check_panel(audit, target, regional, reg_record, corrected, fixed_record, grid, gate['dq_sha256'])
            if historical:
                require(historical['identity']['inputs']['regional_marker']['sha256'] == reg_record['complete_sha256'], 'Historical ASMC regional dependency changed')
                with np.load(audit.path(history['results']), allow_pickle=False) as z, np.load(audit.path(fixed_record['results']), allow_pickle=False) as fresh:
                    for key in ('positions', 'pairs', 'subsets'):
                        require(np.array_equal(z[key], fresh[key]), f'Historical/corrected observed panel differs: {suffix}/{key}')
                    require(z['mean'].shape == fresh['mean'].shape and np.isfinite(z['mean']).all() and np.all(z['mean'] > 0), 'Invalid historical ASMC product')
            registry['replacements'][suffix] = dict(target=checked, old_task_id=task['id'], new_task_id=replacement['id'],
                                                    original=history, regional_dependency=reg_record, corrected=fixed_record)
        require(len(registry['replacements']) == 36, 'Incomplete replacement inventory')
        # Only after all36 have validated do we materialize the corrected view.
        for suffix, entry in registry['replacements'].items():
            destination = output/'asmc'/suffix
            destination.mkdir(parents=True)
            selected = {}
            for label, digest_key in [('marker', 'complete_sha256'), ('results', 'results_sha256')]:
                source = audit.path(entry['corrected'][label])
                target = destination/source.name
                shutil.copyfile(source, target)
                require(sha(target) == entry['corrected'][digest_key], f'Artifact changed during collection: {suffix}')
                selected[str(target.relative_to(output))] = dict(sha256=sha(target), bytes=target.stat().st_size)
            entry['selected_artifacts'] = selected
        registry['complete'] = True
    except Exception as exc:
        registry['errors'].append(f'{type(exc).__name__}: {exc}')
    registry['verified_input_artifacts'] = audit.artifacts
    write(output/'replacement_registry.json', registry)
    write(output/'validation.json', dict(passed=registry['complete'], validated_replacements=len(registry['replacements']),
                                        selected_replacements=sum('selected_artifacts' in entry for entry in registry['replacements'].values()),
                                        expected_replacements=36, errors=registry['errors'],
                                        registry_sha256=sha(output/'replacement_registry.json')))
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='One already-synchronized host root')
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--replacement-manifest', required=True)
    parser.add_argument('--output-dir', required=True, help='New separate immutable collection directory')
    args = parser.parse_args()
    result = collect(args.root, args.original_manifest, args.replacement_manifest, args.output_dir)
    print(json.dumps({'complete': result['complete'], 'replacements': len(result['replacements']), 'errors': result['errors']}))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
