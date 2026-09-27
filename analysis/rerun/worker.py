"""Execute immutable JSON task manifests on one GPU or CPU allocation.

Tasks use argument arrays, isolated attempt folders, advisory process locks and
hashed completion records. Only successful, validated tasks can be collected.
No old analysis output is ever used as a completion signal.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


class FrozenSourceError(RuntimeError):
    pass


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def inside(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root):raise ValueError(f'Path escapes campaign/source root: {path}')
    return path


def task_values(root,source,tid):
    return dict(root=str(root),source=str(source),inputs=str(root/'inputs'),tools=str(root/'inputs/tools'),
                python=sys.executable,output=str(root/'outputs/tasks'/tid/'attempt-001'))


def artifact_path(name,out,values):
    return inside(Path(out)/name.format(**values),values['root'])


def verify_source(source,source_hash,manifest,manifest_hash):
    """Recheck before dispatch and before recording completion, not just once."""
    if sha(manifest)!=manifest_hash:raise FrozenSourceError('Task manifest changed during worker execution')
    raw=(source/'SOURCE_MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source_hash:raise FrozenSourceError('Task/source manifest mismatch or source manifest changed')
    frozen=json.loads(raw)
    if not re.fullmatch(r'[0-9a-fA-F]{40}',frozen.get('git_commit','')):
        raise FrozenSourceError('Source manifest must pin a full Git commit')
    if not isinstance(frozen.get('files'),dict) or not frozen['files']:
        raise FrozenSourceError('Source manifest must list nonempty file hashes')
    for name,entry in frozen['files'].items():
        path=inside(source/name,source)
        digest=entry.get('sha256') if isinstance(entry,dict) else entry
        if not path.is_file() or sha(path)!=digest:raise FrozenSourceError(f'Source changed: {name}')
    return frozen


def validate_marker(path,root,configuration_hash=None):
    """Check scientific success markers, not merely their presence/nonzero size."""
    name=path.name
    if name.endswith('.metadata.json'):
        d=json.loads(path.read_text())
        if not d.get('complete') or d.get('calibration',{}).get('time_units')!='generations':
            raise ValueError(f'Incomplete/uncalibrated output {path}')
        for suffix,digest in d['output_sha256'].items():
            q=inside(path.with_name(name.removesuffix('.metadata.json')+suffix),root)
            if sha(q)!=digest:raise ValueError(f'Output digest mismatch {q}')
    elif name=='COMPLETE.json':
        d=json.loads(path.read_text())
        if d.get('complete') is not True:raise ValueError(f'Incomplete output {path}')
        for relative,digest in d['outputs'].items():
            q=inside(path.parent/relative,path.parent)
            if not q.is_file() or sha(q)!=digest:raise ValueError(f'Output digest mismatch {q}')
    elif name=='DONE.json':
        d=json.loads(path.read_text())
        if d.get('status')!='complete':raise ValueError(f'Failed external workflow {path}')
        if configuration_hash and d.get('manifest_sha256')!=configuration_hash:
            raise ValueError(f'External workflow manifest mismatch {path}')
    elif name=='validation.json':
        if json.loads(path.read_text()).get('passed') is not True:raise ValueError(f'Failed validation gate {path}')
    elif name=='result.json':
        if json.loads(path.read_text()).get('status')!='ok':raise ValueError(f'Failed result {path}')


def validate(task,out,values):
    files={}
    for name in task['expected']:
        path=artifact_path(name,out,values)
        if not path.is_file() or not path.stat().st_size:raise ValueError(f'Missing output {path}')
        validate_marker(path,values['root'],task.get('configuration_manifest_sha256'))
        # Keep portable template keys: the collector maps them onto each host.
        files[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
    return files


def completion(state,tid,source_hash,manifest_hash):
    path=state/(tid+'.done.json')
    if not path.is_file():return None
    d=json.loads(path.read_text())
    if (d.get('task_id')!=tid or d.get('source_manifest_sha256')!=source_hash
            or d.get('task_manifest_sha256')!=manifest_hash or d.get('returncode')!=0
            or d.get('error') or not isinstance(d.get('outputs'),dict)):
        raise ValueError(f'Existing completion belongs to a different/invalid task manifest: {path}')
    return d


def dependencies_ready(task,values,producers,state,source_hash,manifest_hash,root,source,gates=()):
    for requirement in task.get('requires',[]):
        path=Path(requirement.format(**values))
        if not path.is_absolute():path=Path(values['output'])/path
        path=path.resolve()
        if not path.exists():return False
        if str(path) in gates or path.name.lower() in ('production-approved.json','production_gate.json'):
            gate=json.loads(path.read_text())
            if gate.get('passed') is not True or gate.get('source_manifest_sha256')!=source_hash:
                raise ValueError(f'Production gate must pass for this exact source manifest: {path}')
        producer=producers.get(str(path))
        if producer:
            tid,name=producer
            if (state/(tid+'.failed.json')).exists():return False
            done=completion(state,tid,source_hash,manifest_hash)
            if done is None:return False
            detail=done['outputs'].get(name)
            if not detail or not path.is_file() or path.stat().st_size!=detail['bytes'] or sha(path)!=detail['sha256']:
                raise ValueError(f'Completed dependency output changed: {path}')
        # External data files may be staged without a producer task. Scientific
        # completion/gate markers still must certify success before dispatch.
        if path.is_file():validate_marker(path,root)
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--kind',choices=['gpu','cpu'],required=True)
    p.add_argument('--once',action='store_true')
    p.add_argument('--idle-exit-seconds',type=float,default=300)
    p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--max-start-seconds',type=float,default=0,
                   help='Stop starting tasks after this duration; finish the active task. 0 means unlimited.')
    p.add_argument('--only',nargs='*')
    p.add_argument('--include-dedicated',action='store_true')
    a=p.parse_args()
    if a.idle_exit_seconds<0 or a.poll_seconds<=0 or a.max_start_seconds<0:
        p.error('Nonnegative idle/start budgets and positive polling interval required')
    started=time.monotonic()
    budget_exhausted=lambda: a.max_start_seconds>0 and time.monotonic()-started>=a.max_start_seconds
    root=Path(a.root).resolve();manifest=Path(a.manifest).resolve()
    raw=manifest.read_bytes();spec=json.loads(raw);manifest_hash=hashlib.sha256(raw).hexdigest()
    source=inside(root/spec.get('source','source'),root)
    source_hash=spec['source_manifest_sha256']
    frozen=verify_source(source,source_hash,manifest,manifest_hash)
    all_tasks=spec['tasks'];ids=set();producers={}
    for task in all_tasks:
        tid=task['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',tid) or tid in ids:raise ValueError(f'Unsafe/duplicate task ID: {tid}')
        ids.add(tid)
        if task['kind'] not in ('gpu','cpu') or not task.get('stage'):raise ValueError(f'Invalid task kind/stage: {tid}')
        if (not isinstance(task.get('command'),list) or not task['command']
                or not isinstance(task.get('expected'),list) or not task['expected']
                or not all(isinstance(x,str) for x in task['command']+task['expected']+task.get('requires',[]))):
            raise ValueError(f'Task needs argument-array command and nonempty expected files: {tid}')
        values=task_values(root,source,tid)
        for token in task['command']+task.get('requires',[]):token.format(**values)
        for name in task['expected']:
            path=str(artifact_path(name,values['output'],values))
            if path in producers:raise ValueError(f'Duplicate expected artifact: {path}')
            producers[path]=(tid,name)
    if a.only and not set(a.only)<=ids:raise ValueError('Unknown task ID in --only')
    tasks=[t for t in all_tasks if t['kind']==a.kind and (not a.only or t['id'] in a.only)
           and (a.include_dedicated or not t.get('resources',{}).get('dedicated'))]
    if not tasks:raise ValueError('No matching tasks')
    gates=set()
    if spec.get('validation_gate'):
        gates.add(str(Path(spec['validation_gate'].format(**task_values(root,source,tasks[0]['id']))).resolve()))
    state=root/'state';state.mkdir(exist_ok=True)
    idle=time.monotonic()
    while True:
        ran=False;pending=False
        for task in tasks:
            if budget_exhausted():return
            tid=task['id'];values=task_values(root,source,tid)
            done=state/(tid+'.done.json');failed=state/(tid+'.failed.json');adopted=state/(tid+'.adopted.json')
            if completion(state,tid,source_hash,manifest_hash) is not None or failed.exists() or adopted.exists():continue
            pending=True
            if not dependencies_ready(task,values,producers,state,source_hash,manifest_hash,root,source,gates):continue
            with open(state/(tid+'.lock'),'a+') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if completion(state,tid,source_hash,manifest_hash) is not None or failed.exists() or adopted.exists():continue
                # A producer or adoption marker may have changed while waiting.
                if not dependencies_ready(task,values,producers,state,source_hash,manifest_hash,root,source,gates):continue
                if budget_exhausted():return
                out=Path(values['output']);cmd=[x.format(**values) for x in task['command']]
                record=dict(task_id=tid,stage=task['stage'],hostname=socket.gethostname(),command=cmd,
                            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),started_epoch=time.time(),
                            task_manifest_sha256=manifest_hash,source_manifest_sha256=source_hash,
                            git_commit=frozen['git_commit'],output=str(out),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                            slurm_job_id=os.environ.get('SLURM_JOB_ID'))
                try:
                    verify_source(source,source_hash,manifest,manifest_hash)
                    configuration_hash=task.get('configuration_manifest_sha256')
                    if configuration_hash:
                        config_path=Path(cmd[cmd.index('--manifest')+1])
                        if sha(config_path)!=configuration_hash:raise ValueError(f'Configuration manifest changed: {config_path}')
                    if out.exists():raise FileExistsError(f'Prior incomplete attempt requires inspection: {out}')
                    out.mkdir(parents=True)
                    atomic_json(state/(tid+'.running.json'),record)
                    print(json.dumps({'starting':tid,'host':socket.gethostname(),'command':cmd}),flush=True)
                    with open(out/'run.log','w') as log:
                        process=subprocess.run(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
                    record['returncode']=process.returncode
                    if process.returncode:raise RuntimeError(f'Process exit {process.returncode}; see {out}/run.log')
                    verify_source(source,source_hash,manifest,manifest_hash)
                    record['outputs']=validate(task,out,values)
                    record['completed_epoch']=time.time();record['seconds']=record['completed_epoch']-record['started_epoch']
                    atomic_json(done,record)
                except Exception as exc:
                    record['error']=str(exc);record['failed_epoch']=time.time();atomic_json(failed,record)
                    print(json.dumps({'failed':tid,'error':str(exc)}),flush=True)
                    if isinstance(exc,FrozenSourceError):raise
                finally:
                    (state/(tid+'.running.json')).unlink(missing_ok=True)
                ran=True;idle=time.monotonic()
                if a.once:return
        if not pending:return
        if not ran:
            remaining=a.idle_exit_seconds-(time.monotonic()-idle)
            if a.max_start_seconds>0:remaining=min(remaining,a.max_start_seconds-(time.monotonic()-started))
            if remaining<=0:return
            time.sleep(min(a.poll_seconds,remaining))


if __name__=='__main__':
    main()
