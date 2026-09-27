from pathlib import Path
import importlib.util
import json
import pytest

FILE = Path(__file__).resolve().parents[1] / 'make_tasks.py'
spec = importlib.util.spec_from_file_location('make_tasks', FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def arguments(tmp_path, *args):
    return m.parser().parse_args(['--source-manifest-sha256', 'a' * 64,
                                  '--output', str(tmp_path / 'tasks.json'), *args])


def test_complete_orthogonal_inventory_and_dependencies(tmp_path):
    result = m.build(arguments(tmp_path))
    tasks = {t['id']: t for t in result['tasks']}
    assert len(tasks) == 1433
    assert sum(t['kind'] == 'gpu' for t in tasks.values()) == 36
    normalization = tasks['orthogonal-normalize-YRI']
    assert len([p for p in normalization['requires'] if p.endswith('COMPLETE.json')]) == 22
    assert tasks['orthogonal-asmc-TREM2_YRI']['requires'][-2].endswith('/regional/TREM2_YRI/COMPLETE.json')
    assert tasks['orthogonal-regional-IFIH1_IBS']['kind'] == 'gpu'
    # Every output/command resolves using the worker's documented values.
    values = dict(root='/run', source='/run/source', inputs='/run/inputs', tools='/run/inputs/tools', python='/env/python', output='/run/task')
    for task in tasks.values():
        for field in ('command', 'requires', 'expected'):
            for token in task[field]:
                token.format(**values)
        assert all(p.startswith('{root}/outputs/orthogonal/') for p in task['expected'])


def test_fresh_relate_inventory_and_dedicated_resources(tmp_path):
    rel = m.module('relate_clues')
    loci = {name: dict(group=g, population=p) for name, g, p, *_ in rel.LOCI}
    groups = {name: {} for name in rel.GROUPS}
    inventory = dict(threads=16, relate_memory_gb=128, groups=groups, loci=loci,
                     tasks=dict(prepare=list(groups), relate=list(groups),
                                popsize=sorted({v['group'] + '_' + v['population'] for v in loci.values()}),
                                locus=list(loci)))
    path = tmp_path / 'relate.json'
    path.write_text(json.dumps(inventory))
    result = m.build(arguments(tmp_path, '--relate-manifest', str(path)))
    tasks = {t['id']: t for t in result['tasks']}
    assert len(tasks) == 1469  # 1433 orthogonal +5+5+7+19 Relate/CLUES
    assert result['summary']['dedicated_tasks'] == 36
    trem = tasks['relate-locus-TREM2_41121942']
    assert trem['resources']['memory_gb'] == 512
    assert trem['resources']['dedicated'] is True
    assert any(p.endswith('/popsize/chr6_EUREAS_IBS/DONE.json') for p in trem['requires'])
    assert tasks['relate-popsize-chr12_EAS_CDX']['resources']['cpus'] == 16


def test_merge_relative_outputs_and_conflicting_source(tmp_path):
    base = tmp_path / 'base.json'
    tasks = [dict(id=f'core-{i}', kind='gpu', stage='core', command=['x'], expected=['result.json']) for i in range(2)]
    base.write_text(json.dumps(dict(source='source', source_manifest_sha256='a' * 64, tasks=tasks)))
    result = m.build(arguments(tmp_path, '--skip-orthogonal', '--merge-task-manifest', str(base)))
    assert len(result['tasks']) == 2
    base.write_text(base.read_text().replace('a' * 64, 'b' * 64))
    with pytest.raises(ValueError, match='identical frozen source'):
        m.build(arguments(tmp_path, '--merge-task-manifest', str(base)))


def test_duplicate_outputs_and_dependency_cycles_fail():
    with pytest.raises(ValueError, match='same expected output'):
        m.validate_tasks([dict(id='a', expected=['{root}/same']), dict(id='b', expected=['{root}/same'])])
    with pytest.raises(ValueError, match='cycle'):
        m.validate_tasks([dict(id='a', expected=['{root}/a'], requires=['{root}/b']),
                          dict(id='b', expected=['{root}/b'], requires=['{root}/a'])])
