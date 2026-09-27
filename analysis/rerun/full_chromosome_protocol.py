"""Immutable whole-chromosome genome task protocol, without changing inference.

Generation does not approve or execute tasks. Collection requires a separate
passed cross-host gate explicitly bound to this protocol and task manifest.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = 'full_chromosome_v1'
SOURCE = 'source-v2'
MANIFEST_PATH = 'control/tasks-genome-full.json'
GATE = '{root}/validation/full-chromosome-approved.json'
POPS = sorted('ACB ASW ESN GWD LWK MSL YRI CEU FIN GBR IBS TSI CDX CHB CHS JPT KHV BEB GIH ITU PJL STU CLM MXL PEL PUR'.split())
SETTINGS = dict(mu=1.25e-8, rho=1e-8, core_block_sites=6000000, flank_sites=0,
                pair_chunk=512, lead_half_bp=25000, time_units='generations',
                positions_base=1, positions_dtype='int64',
                inference_context='one full-chromosome site block; automatic native pair batching',
                sequence_length_convention='last retained VCF POS + 1',
                pair_selection='all unordered focal-population haplotype pairs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def task_id(chrom, pop):
    return f'genome-full_chr{chrom}_{pop}'


def definition(chrom, pop):
    """The exact portable scientific command; resource requests are separate."""
    return dict(id=task_id(chrom, pop), kind='gpu', stage='genome_full',
        command=['{python}', '{source}/analysis/genome_wide/infer_chromosome.py',
                 '--chr', str(chrom), '--populations', pop,
                 '--cache-dir', '{inputs}/cache', '--samples', '{inputs}/samples.txt',
                 '--output-dir', '{output}/results', '--core-block-sites', '6000000',
                 '--flank-sites', '0', '--pair-chunk', '512', '--mu', '1.25e-08',
                 '--rho', '1e-08', '--lead-variants', '{inputs}/akbari_lead_variants_grch38.tsv',
                 '--lead-half-bp', '25000'],
        requires=[f'{{inputs}}/cache/parsed/chr{chrom}/READY.json',
                  f'{{inputs}}/cache/genes/chr{chrom}_genes.tsv', '{inputs}/samples.txt',
                  '{inputs}/akbari_lead_variants_grch38.tsv', GATE],
        expected=[f'results/chr{chrom}/{pop}{ext}' for ext in ('.npz', '.csv', '.metadata.json')])


def validate_manifest(manifest):
    require(manifest.get('schema') == 1 and manifest.get('protocol') == PROTOCOL
            and manifest.get('source') == SOURCE, 'Not the full-chromosome source-v2 protocol')
    require(manifest.get('scientific_settings') == SETTINGS, 'Full-chromosome scientific settings changed')
    require(manifest.get('validation_gate') == GATE
            and manifest.get('task_manifest_relative_path') == MANIFEST_PATH,
            'Full-chromosome gate/manifest path changed')
    require(re.fullmatch(r'[0-9a-f]{64}', manifest.get('source_manifest_sha256', '')),
            'Full protocol requires an exact source-manifest hash')
    tasks = manifest.get('tasks', [])
    require(len(tasks) == 572, 'Full protocol requires exactly 572 tasks')
    by_id = {t['id']: t for t in tasks}
    require(len(by_id) == 572, 'Duplicate full-chromosome task IDs')
    mapping = {}
    for chrom in range(1, 23):
        for pop in POPS:
            canonical = definition(chrom, pop)
            task = by_id.get(canonical['id'])
            require(task is not None and set(task) == set(canonical) | {'resources'},
                    f'Missing/unexpected full task: {canonical["id"]}')
            require(all(task[key] == value for key, value in canonical.items()),
                    f'Full task command/settings/output mapping changed: {canonical["id"]}')
            mapping[chrom, pop] = task
    return mapping


def generate(original_manifest, source_manifest):
    old, frozen = read(original_manifest), read(source_manifest)
    source_hash = sha(source_manifest)
    require(old.get('source') == SOURCE and old.get('source_manifest_sha256') == source_hash,
            'Original task manifest does not match unchanged source-v2')
    require(re.fullmatch(r'[0-9a-fA-F]{40}', frozen.get('git_commit', '')) and frozen.get('files'),
            'Source manifest must contain full commit and file hashes')
    old_tasks = old.get('tasks', [])
    expected = {f'genome_chr{chrom}_{pop}': (chrom, pop) for chrom in range(1, 23) for pop in POPS}
    require(len(old_tasks) == 572 and {t['id'] for t in old_tasks} == set(expected),
            'Original genome inventory must contain exactly 572 chromosome/population tasks')
    tasks = []
    for old_task in old_tasks:
        chrom, pop = expected[old_task['id']]
        task = definition(chrom, pop)
        task['resources'] = copy.deepcopy(old_task['resources'])
        tasks.append(task)
    result = dict(schema=1, protocol=PROTOCOL, source=SOURCE, source_manifest_sha256=source_hash,
                  task_manifest_relative_path=MANIFEST_PATH, scientific_settings=SETTINGS.copy(),
                  validation_gate=GATE, tasks=tasks,
                  supersedes=dict(original_task_manifest_sha256=sha(original_manifest),
                                  reason='Eliminate internal finite-flank site boundaries; preserve old outputs.'),
                  collection=dict(expected_gene_results=572, cross_host_validation_required=True,
                                  new_registry_required=True),
                  execution_note='One worker per allocated GPU initially. Raw cache sites must not exceed 6000000; flank 0 fails instead of silently splitting a larger retained chromosome.')
    validate_manifest(result)
    return result


class FullProtocol:
    def __init__(self, manifest_path, hosts, gate, source_hash, source_directory):
        require(str(source_directory) == SOURCE, 'Full collection requires unchanged source-v2')
        self.manifest = read(manifest_path)
        self.manifest_sha256 = sha(manifest_path)
        self.tasks = validate_manifest(self.manifest)
        require(self.manifest['source_manifest_sha256'] == source_hash, 'Full manifest/source mismatch')
        require(gate.get('passed') is True and gate.get('protocol') == PROTOCOL
                and gate.get('task_manifest_sha256') == self.manifest_sha256
                and gate.get('source_manifest_sha256') == source_hash
                and gate.get('scientific_settings') == SETTINGS
                and gate.get('cross_host_parity_validated') is True,
                'Full collection requires an explicit passed full-protocol cross-host gate')
        tested = gate.get('validated_hosts', [])
        require(isinstance(tested, list) and len(tested) >= 2 and len(tested) == len(set(tested))
                and set(hosts) <= set(tested), 'Full cross-host gate does not cover selected hosts')
        for root in hosts.values():
            require(sha(Path(root)/MANIFEST_PATH) == self.manifest_sha256,
                    'Synchronized host task manifest differs from full collection manifest')

    def check_receipt(self, done, task):
        tid = task['id']
        require(done.get('task_manifest_sha256') == self.manifest_sha256
                and done.get('stage') == task['stage'], f'Full task receipt manifest/stage mismatch: {tid}')
        output = Path(done['output'])
        require(output.is_absolute() and output.parts[-4:] == ('outputs', 'tasks', tid, 'attempt-001'),
                f'Unexpected full task attempt path: {tid}')
        root = output.parents[3]
        require(isinstance(done.get('command'), list) and len(done['command']) > 1
                and Path(done['command'][0]).is_absolute(), f'Invalid full task executable: {tid}')
        values = dict(root=str(root), source=str(root/SOURCE), inputs=str(root/'inputs'),
                      tools=str(root/'inputs/tools'), output=str(output), python=done['command'][0])
        require(done['command'] == [s.format(**values) for s in task['command']],
                f'Executed full task command differs from manifest: {tid}')
        require(set(done.get('outputs', {})) == set(task['expected']),
                f'Full task output inventory differs from manifest: {tid}')
        return values

    def check_metadata(self, metadata, done, task):
        values = self.check_receipt(done, task)
        command = task['command']
        flags = dict(zip(command[2::2], command[3::2]))
        expected = {k.removeprefix('--').replace('-', '_'): v.format(**values)
                    for k, v in flags.items() if k != '--populations'}
        for key in ('chr', 'core_block_sites', 'flank_sites', 'pair_chunk', 'lead_half_bp'):
            expected[key] = int(expected[key])
        for key in ('mu', 'rho'):
            expected[key] = float(expected[key])
        expected['sequence_length'] = None
        require(metadata['identity']['configuration'] == expected,
                f'Full task metadata configuration differs from exact command: {task["id"]}')
        require(set(metadata['identity']['source_sha256']) == {
                    'analysis/genome_wide/infer_chromosome.py', 'analysis/genome_wide/rerun_support.py'},
                'Full inference metadata is missing its actual source files')
        calibration = metadata['calibration']
        require(calibration.get('time_units') == 'generations'
                and calibration.get('physical_mu') == SETTINGS['mu']
                and calibration.get('physical_rho') == SETTINGS['rho'],
                'Full task calibration differs from physical rate protocol')
        shape = metadata['identity']['inputs']['cache']['record']['shape']
        require(len(shape) == 2 and 0 < shape[1] <= SETTINGS['core_block_sites'],
                'Raw chromosome exceeds the whole-chromosome core bound')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = generate(args.original_manifest, args.source_manifest)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(dict(tasks=len(result['tasks']), protocol=PROTOCOL,
                          task_manifest_sha256=sha(args.output), output=args.output)))


if __name__ == '__main__':
    main()
