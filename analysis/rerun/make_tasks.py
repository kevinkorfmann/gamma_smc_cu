#!/usr/bin/env python3
"""Build the worker DAG for complete orthogonal and fresh Relate/CLUES reruns.

Templates remain portable between hosts. Create each host's Relate manifest
first with relate_clues.py manifest, then supply its inventory here. No task
or scientific computation is executed by this generator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location('task_inventory_' + name, HERE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Unsafe task identifier: {value!r}')
    return value


def resources(cpus, memory_gb, walltime_hours, dedicated=False):
    return dict(cpus=cpus, memory_gb=memory_gb, walltime_hours=walltime_hours, dedicated=dedicated)


def orthogonal_tasks(a):
    orth = module('orthogonal')
    inventory, _ = orth.task_table(SimpleNamespace(targets=a.targets))
    output = a.orthogonal_output.rstrip('/')
    base = ['{python}', '{source}/analysis/rerun/orthogonal.py']
    common = ['--inputs-root', '{inputs}', '--output-root', output, '--tools-root', '{tools}',
              '--threads', str(a.orthogonal_threads), '--core-block-sites', str(a.core_block_sites),
              '--flank-sites', str(a.flank_sites), '--window-sites', str(a.window_sites),
              '--step-sites', str(a.step_sites), '--pair-cap', str(a.pair_cap),
              '--mu', str(a.mu), '--rho', str(a.rho)]
    tasks = []
    for row in inventory.to_dict('records'):
        action, chrom, pop, gene = row['action'], row['chr'], row['pop'], row['gene']
        command = base + [action] + common
        requires = ['{inputs}/samples.txt']
        if chrom != '':
            chrom = int(chrom)
            command += ['--chr', str(chrom)]
            requires += [f'{{inputs}}/cache/parsed/chr{chrom}/READY.json',
                         f'{{inputs}}/cache/genes/chr{chrom}_genes.tsv']
        command += ['--pop', pop]
        if gene:
            safe_id(gene)
            command += ['--gene', gene]
        if action in ('normalize', 'h12-aggregate', 'neighborhood-aggregate'):
            key = pop
            dependency = {'normalize': 'selscan', 'h12-aggregate': 'h12',
                          'neighborhood-aggregate': 'neighborhood-background'}[action]
            requires += [f'{output}/{dependency}/chr{c}_{pop}/COMPLETE.json' for c in range(1, 23)]
            if action == 'normalize':
                requires += [f'{{inputs}}/cache/genes/chr{c}_genes.tsv' for c in range(1, 23)]
        else:
            key = f'{gene}_{pop}' if gene else f'chr{chrom}_{pop}'
        if action == 'asmc':
            command += ['--decoding-quantities', '{tools}/asmc_data/CEU_csfs50/decoding.decodingQuantities.gz']
            requires += [f'{output}/regional/{gene}_{pop}/COMPLETE.json',
                         '{tools}/asmc_data/CEU_csfs50/decoding.decodingQuantities.gz']
        if action == 'selscan':
            requires += ['{tools}/selscan/bin/linux/selscan']
        kind = 'gpu' if action == 'regional' else 'cpu'
        memory = 64 if action in ('normalize', 'regional') else 32
        cpus = a.orthogonal_threads if action in ('selscan', 'regional') else 1
        tasks.append(dict(id=safe_id(f'orthogonal-{action}-{key}'), kind=kind,
                          stage='orthogonal_' + action.replace('-', '_'), command=command,
                          requires=requires, expected=[f'{output}/{action}/{key}/COMPLETE.json'],
                          resources=resources(cpus, memory, 24)))
    return tasks


def relate_tasks(path, output, locus_memory_gb=512):
    manifest = json.loads(Path(path).read_text())
    needed = {'prepare', 'relate', 'popsize', 'locus'}
    if set(manifest.get('tasks', {})) != needed:
        raise ValueError('Relate manifest must have prepare, relate, popsize and locus task inventories')
    groups = set(manifest['groups'])
    loci = manifest['loci']
    expected_popsize = {locus['group'] + '_' + locus['population'] for locus in loci.values()}
    if (set(manifest['tasks']['prepare']) != groups or set(manifest['tasks']['relate']) != groups
            or set(manifest['tasks']['popsize']) != expected_popsize
            or set(manifest['tasks']['locus']) != set(loci)
            or any(locus['group'] not in groups for locus in loci.values())):
        raise ValueError('Relate task inventory is incomplete or inconsistent with groups/loci')
    output = output.rstrip('/')
    marker = lambda stage, task: f'{output}/state/{stage}/{task}/DONE.json'
    threads, memory = int(manifest['threads']), int(manifest['relate_memory_gb'])
    tasks = []
    for stage in ('prepare', 'relate', 'popsize', 'locus'):
        for task in manifest['tasks'][stage]:
            safe_id(task)
            requires = [output + '/manifest.json']
            if stage == 'prepare':
                if task not in manifest['groups']:
                    raise ValueError(f'Unknown Relate group {task}')
            elif stage == 'relate':
                requires += [marker('prepare', task)]
            elif stage == 'popsize':
                group, _ = task.rsplit('_', 1)
                requires += [marker('prepare', group), marker('relate', group)]
            elif stage == 'locus':
                locus = manifest['loci'][task]
                group, pop = locus['group'], locus['population']
                requires += [marker('prepare', group), marker('popsize', group + '_' + pop)]
            cpus = threads if stage == 'popsize' else (4 if stage == 'prepare' else 1)
            # Relate's internal memory limit needs headroom; CLUES arrays and
            # chromosome conversion also require dedicated large-memory jobs.
            mem = max(locus_memory_gb, memory + 64) if stage == 'locus' else max(192, memory + 64)
            hours = 72 if stage in ('relate', 'locus') else 48
            tasks.append(dict(id=safe_id(f'relate-{stage}-{task}'), kind='cpu',
                stage='relate_' + stage,
                command=['{python}', '{source}/analysis/rerun/relate_clues.py', 'run',
                         '--manifest', output + '/manifest.json', '--stage', stage, '--task', task],
                requires=requires, expected=[marker(stage, task)],
                resources=resources(cpus, mem, hours, dedicated=True),
                configuration_manifest_sha256=digest(path)))
    return tasks


def validate_tasks(tasks):
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs in merged manifest')
    def key(path, tid):
        task_output = '{root}/outputs/tasks/' + tid + '/attempt-001'
        path = path.replace('{output}', task_output)
        return path if path.startswith(('/', '{')) else task_output + '/' + path
    producers = {}
    for task in tasks:
        for name in task['expected']:
            name = key(name, task['id'])
            if name in producers:
                raise ValueError(f'Two tasks write the same expected output: {name}')
            producers[name] = task['id']
    graph = {t['id']: {producers[key(p,t['id'])] for p in t.get('requires', []) if key(p,t['id']) in producers} for t in tasks}
    visiting, visited = set(), set()

    def visit(task):
        if task in visiting:
            raise ValueError(f'Dependency cycle involving {task}')
        if task in visited:
            return
        visiting.add(task)
        for dependency in graph[task]:
            visit(dependency)
        visiting.remove(task)
        visited.add(task)
    for task in graph:
        visit(task)
    return sum(len(deps) for deps in graph.values())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-manifest', help='Frozen SOURCE_MANIFEST.json to hash')
    source.add_argument('--source-manifest-sha256', help='Already verified SHA256 of frozen SOURCE_MANIFEST.json')
    p.add_argument('--source', default='source', help='Source directory relative to the run root')
    p.add_argument('--output', required=True, help='New worker task manifest JSON file')
    p.add_argument('--merge-task-manifest', action='append', default=[], help='Existing worker manifest to merge, e.g. core GPU tasks')
    p.add_argument('--skip-orthogonal', action='store_true')
    p.add_argument('--targets', help='Custom orthogonal target TSV used to construct the inventory')
    p.add_argument('--orthogonal-output', default='{root}/outputs/orthogonal')
    p.add_argument('--orthogonal-threads', type=int, default=4)
    p.add_argument('--core-block-sites', type=int, default=65536)
    p.add_argument('--flank-sites', type=int, default=8192)
    p.add_argument('--window-sites', type=int, default=400)
    p.add_argument('--step-sites', type=int, default=50)
    p.add_argument('--pair-cap', type=int, default=20)
    p.add_argument('--mu', type=float, default=1.25e-8)
    p.add_argument('--rho', type=float, default=1e-8)
    p.add_argument('--relate-manifest', help='Fresh host-specific relate_clues.py manifest.json inventory')
    p.add_argument('--relate-output', default='{root}/outputs/relate', help='Portable location of that fresh Relate manifest/output tree')
    p.add_argument('--relate-locus-memory-gb', type=int, default=512, help='Dedicated CLUES memory request; historical requirements ranged from256 to900GB')
    return p


def build(a):
    source_hash = digest(a.source_manifest) if a.source_manifest else a.source_manifest_sha256
    if not re.fullmatch('[0-9a-f]{64}', source_hash):
        raise ValueError('Invalid source-manifest SHA256')
    if Path(a.source).is_absolute() or '..' in Path(a.source).parts:
        raise ValueError('--source must be a safe relative path')
    if min(a.orthogonal_threads, a.core_block_sites, a.window_sites, a.step_sites, a.pair_cap, a.relate_locus_memory_gb) < 1 or a.flank_sites < 0 or a.mu <= 0 or a.rho < 0:
        raise ValueError('Invalid analysis parameters')
    tasks = [] if a.skip_orthogonal else orthogonal_tasks(a)
    provenance = {}
    if a.targets:
        provenance['orthogonal_targets_sha256'] = digest(a.targets)
    if a.relate_manifest:
        tasks += relate_tasks(a.relate_manifest, a.relate_output, a.relate_locus_memory_gb)
        provenance['relate_manifest_sha256'] = digest(a.relate_manifest)
    for path in a.merge_task_manifest:
        other = json.loads(Path(path).read_text())
        if other['source_manifest_sha256'] != source_hash or other.get('source', 'source') != a.source:
            raise ValueError('Merged manifests must refer to the identical frozen source')
        tasks += other['tasks']
    if not tasks:
        raise ValueError('No tasks requested')
    edges = validate_tasks(tasks)
    return dict(schema=1, source=a.source, source_manifest_sha256=source_hash,
                provenance=provenance, tasks=tasks,
                summary=dict(tasks=len(tasks), internal_dependency_edges=edges,
                             dedicated_tasks=sum(bool(t.get('resources', {}).get('dedicated')) for t in tasks)),
                scheduling_note='Dedicated tasks require explicitly provisioned CPU jobs; ordinary workers must exclude resources.dedicated=true.')


def main():
    a = parser().parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace worker manifest {output}')
    spec = build(a)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(spec['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
