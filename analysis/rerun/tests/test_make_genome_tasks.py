from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pytest

FILE = Path(__file__).resolve().parents[1] / 'make_genome_tasks.py'
spec = importlib.util.spec_from_file_location('make_genome_tasks', FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_all_572_production_tasks_exact_flags_order_and_gates(tmp_path):
    source = tmp_path / 'SOURCE_MANIFEST.json'
    source.write_text(json.dumps({'files': {'analysis/genome_wide/infer_chromosome.py': 'x'}}))
    args = m.parser().parse_args(['--source-manifest', str(source), '--output', str(tmp_path / 'tasks.json')])
    manifest = m.build(args)
    assert manifest['source'] == 'source-v2'
    assert manifest['source_manifest_sha256'] == m.sha256(source)
    tasks = manifest['tasks']
    assert len(tasks) == len({t['id'] for t in tasks}) == 572
    assert [t['id'] for t in tasks[:3]] == ['genome_chr22_ASW', 'genome_chr22_YRI', 'genome_chr22_CEU']
    assert [tasks[i * 26]['id'] for i in range(6)] == [f'genome_chr{c}_ASW' for c in (22, 11, 6, 12, 20, 2)]
    assert all(t['kind'] == 'gpu' and t['stage'] == 'genome' for t in tasks)
    command = tasks[0]['command']
    expected_flags = {'--core-block-sites': '65536', '--flank-sites': '8192', '--pair-chunk': '512',
                      '--mu': '1.25e-08', '--rho': '1e-08', '--lead-half-bp': '25000',
                      '--output-dir': '{output}/results'}
    for flag, value in expected_flags.items():
        assert command[command.index(flag) + 1] == value
    assert '{root}/validation/production-approved.json' in tasks[0]['requires']
    assert '{inputs}/cache/parsed/chr22/READY.json' in tasks[0]['requires']
    assert tasks[0]['expected'] == [f'results/chr22/ASW{ext}' for ext in ('.npz', '.csv', '.metadata.json')]
    # Read the actual runner's argparse surface; no CUDA import or inference.
    runner = FILE.parents[1] / 'genome_wide/infer_chromosome.py'
    help_text = subprocess.check_output([sys.executable, str(runner), '--help'], text=True)
    assert all(token in help_text for token in command if token.startswith('--'))
    values = dict(root='/run', source='/run/source-v2', inputs='/run/inputs', tools='/run/inputs/tools', python='/env/python', output='/attempt')
    for task in tasks:
        for key in ('command', 'requires', 'expected'):
            for token in task[key]:
                token.format(**values)


def test_reject_incomplete_frozen_source(tmp_path):
    source = tmp_path / 'SOURCE_MANIFEST.json'
    source.write_text('{}')
    args = m.parser().parse_args(['--source-manifest', str(source), '--output', str(tmp_path / 'tasks.json')])
    with pytest.raises(ValueError, match='lacks'):
        m.build(args)
