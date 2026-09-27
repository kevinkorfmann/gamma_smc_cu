"""Local synthetic subprocess checks of the production worker contract."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rerun_worker',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

SCRIPT='''import json, pathlib, sys, time
mode, target, counter = sys.argv[1:4]
with open(counter,'a') as stream: stream.write(mode+'\\n')
if mode=='slow': time.sleep(.25)
if mode=='mutate': pathlib.Path('immutable.txt').write_text('changed')
path=pathlib.Path(target);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({'status':'ok'}))
if mode=='fail': raise SystemExit(7)
'''


def setup(tmp_path):
    root=tmp_path/'campaign';source=root/'source';source.mkdir(parents=True)
    (source/'job.py').write_text(SCRIPT)
    (source/'immutable.txt').write_text('original')
    frozen=dict(git_commit='a'*40,files={p.name:w.sha(p) for p in source.iterdir()})
    (source/'SOURCE_MANIFEST.json').write_text(json.dumps(frozen))
    return root


def task(tid,mode='ok',expected='result/result.json',requires=()):
    target=expected if expected.startswith('{') else '{output}/'+expected
    return dict(id=tid,stage='synthetic',kind='cpu',expected=[expected],requires=list(requires),
                command=['{python}','{source}/job.py',mode,target,'{root}/executions.txt'])


def write_manifest(root,tasks):
    path=root/'tasks.json'
    path.write_text(json.dumps(dict(source='source',source_manifest_sha256=w.sha(root/'source/SOURCE_MANIFEST.json'),tasks=tasks)))
    return path


def command(root):
    return [sys.executable,str(HERE/'worker.py'),'--root',str(root),'--manifest',str(root/'tasks.json'),
            '--kind','cpu','--idle-exit-seconds','0','--poll-seconds','.01']


def run(root,*extra):
    return subprocess.run(command(root)+list(extra),capture_output=True,text=True,timeout=15)


def test_parallel_workers_run_exactly_once_and_records_map_in_collector(tmp_path):
    root=setup(tmp_path);write_manifest(root,[task('one','slow')])
    a=subprocess.Popen(command(root),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    b=subprocess.Popen(command(root),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    for process in (a,b):
        stdout,stderr=process.communicate(timeout=15)
        assert process.returncode==0,(stdout,stderr)
    assert run(root).returncode==0
    assert (root/'executions.txt').read_text().splitlines()==['slow']
    done=json.loads((root/'state/one.done.json').read_text())
    assert done['task_id']=='one' and done['returncode']==0 and done['completed_epoch']>=done['started_epoch']
    assert done['task_manifest_sha256']==w.sha(root/'tasks.json')
    assert done['source_manifest_sha256']==w.sha(root/'source/SOURCE_MANIFEST.json')
    assert not (root/'state/one.running.json').exists()
    spec=importlib.util.spec_from_file_location('worker_collect_contract',HERE/'collect.py')
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    for name,detail in done['outputs'].items():
        mapped=collector.local_artifact(root,done,name)
        assert mapped.is_file() and collector.sha(mapped)==detail['sha256']


def test_dependency_waits_for_validated_producer_not_just_file_presence(tmp_path):
    root=setup(tmp_path)
    target='{root}/outputs/shared/result.json'
    child=task('child',requires=[target]);parent=task('parent',expected=target)
    write_manifest(root,[child,parent])  # deliberately reverse DAG order
    result=run(root)
    assert result.returncode==0,result.stderr
    assert (root/'executions.txt').read_text().splitlines()==['ok','ok']
    parent_done=json.loads((root/'state/parent.done.json').read_text())
    child_done=json.loads((root/'state/child.done.json').read_text())
    assert child_done['started_epoch']>=parent_done['completed_epoch']
    assert target in parent_done['outputs']  # portable template retained


def test_failed_producer_with_existing_output_never_releases_child(tmp_path):
    root=setup(tmp_path);target='{root}/outputs/shared/result.json'
    write_manifest(root,[task('parent','fail',expected=target),task('child',requires=[target])])
    result=run(root)
    assert result.returncode==0,result.stderr
    assert (root/'outputs/shared/result.json').is_file()
    assert (root/'state/parent.failed.json').is_file()
    assert not (root/'state/parent.done.json').exists()
    assert not (root/'outputs/tasks/child').exists()
    assert (root/'executions.txt').read_text().splitlines()==['fail']
    assert run(root).returncode==0  # failure is never implicitly retried
    assert (root/'executions.txt').read_text().splitlines()==['fail']


def test_missing_dependency_is_not_executed(tmp_path):
    root=setup(tmp_path);write_manifest(root,[task('child',requires=['{inputs}/READY.json'])])
    assert run(root).returncode==0
    assert not (root/'executions.txt').exists()


@pytest.mark.parametrize('when',['before','during'])
def test_source_mutation_prevents_completion_and_further_dispatch(tmp_path,when):
    root=setup(tmp_path);write_manifest(root,[task('one','mutate' if when=='during' else 'ok'),task('two')])
    if when=='before':(root/'source/immutable.txt').write_text('changed')
    result=run(root)
    assert result.returncode!=0
    assert not (root/'state/one.done.json').exists()
    assert not (root/'outputs/tasks/two').exists()
    if when=='during':
        record=json.loads((root/'state/one.failed.json').read_text())
        assert 'Source changed' in record['error'] and record['returncode']==0
    else:assert not (root/'executions.txt').exists()


def test_configuration_manifest_failure_is_recorded_and_does_not_abort_other_tasks(tmp_path):
    root=setup(tmp_path);configuration=root/'configuration.json';configuration.write_text('{}')
    bad=task('bad');bad['configuration_manifest_sha256']='0'*64
    bad['command']+=['--manifest',str(configuration)]
    write_manifest(root,[bad,task('good')])
    result=run(root)
    assert result.returncode==0,result.stderr
    failure=json.loads((root/'state/bad.failed.json').read_text())
    assert failure['task_id']=='bad' and 'Configuration manifest changed' in failure['error']
    assert (root/'state/good.done.json').exists()
    assert (root/'executions.txt').read_text().splitlines()==['ok']


def test_changed_task_manifest_never_reuses_old_success(tmp_path):
    root=setup(tmp_path);write_manifest(root,[task('one')]);assert run(root).returncode==0
    write_manifest(root,[task('one','slow')])
    result=run(root)
    assert result.returncode!=0 and 'different/invalid task manifest' in result.stderr
    assert (root/'executions.txt').read_text().splitlines()==['ok']


@pytest.mark.parametrize('name,payload',[
    ('validation.json',{'passed':False}),('COMPLETE.json',{'complete':False,'outputs':{}}),
    ('DONE.json',{'status':'failed'}),('result.json',{'status':'error'}),
])
def test_failure_markers_cannot_count_as_outputs(tmp_path,name,payload):
    marker=tmp_path/name;marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError):w.validate_marker(marker,tmp_path)


def test_complete_marker_checks_referenced_artifact_hash(tmp_path):
    artifact=tmp_path/'data.txt';artifact.write_text('original')
    marker=tmp_path/'COMPLETE.json';marker.write_text(json.dumps({'complete':True,'outputs':{'data.txt':w.sha(artifact)}}))
    w.validate_marker(marker,tmp_path)
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='digest mismatch'):w.validate_marker(marker,tmp_path)


def test_start_budget_finishes_active_task_but_does_not_start_next(tmp_path):
    root=setup(tmp_path);write_manifest(root,[task('one','slow'),task('two')])
    result=run(root,'--max-start-seconds','.1')
    assert result.returncode==0,result.stderr
    assert (root/'state/one.done.json').exists()
    assert not (root/'outputs/tasks/two').exists()
    assert (root/'executions.txt').read_text().splitlines()==['slow']


@pytest.mark.parametrize('passed,matching',[(False,True),(True,False)])
def test_production_gate_requires_pass_and_matching_source(tmp_path,passed,matching):
    root=setup(tmp_path)
    gate=root/'validation/production-approved.json';gate.parent.mkdir()
    gate.write_text(json.dumps({'passed':passed,'source_manifest_sha256':w.sha(root/'source/SOURCE_MANIFEST.json') if matching else '0'*64}))
    write_manifest(root,[task('one',requires=['{root}/validation/production-approved.json'])])
    result=run(root)
    assert result.returncode!=0 and 'Production gate' in result.stderr
    assert not (root/'executions.txt').exists()
