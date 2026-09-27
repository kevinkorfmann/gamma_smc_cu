"""Generate36 full-context regional tasks and associate existing seeded ASMC.

No computation is launched. Original products and their dependency identities
remain intact; association never rewrites a COMPLETE marker or posterior array.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_regional_asmc as c

OUTPUT = '{root}/outputs/orthogonal-full'
GATE = '{root}/validation/regional-full-approved.json'


def transform(task, gate=GATE):
    f = c.flags(task)
    suffix = f['--gene']+'_'+f['--pop']
    c.require(task['id'] == 'orthogonal-regional-'+suffix and task['stage'] == 'orthogonal_regional'
              and task['command'][:3] == ['{python}','{source}/analysis/rerun/orthogonal.py','regional'], 'Unexpected original regional definition')
    c.require(f['--output-root'] == '{root}/outputs/orthogonal' and f['--pair-cap'] == '20'
              and float(f['--mu']) == 1.25e-8 and float(f['--rho']) == 1e-8, 'Unexpected original scientific settings')
    c.require(f['--core-block-sites'] == '65536' and f['--flank-sites'] == '8192', 'Unexpected original block settings')
    result = copy.deepcopy(task)
    result['id'] = 'orthogonal-full-regional-'+suffix
    result['stage'] = 'orthogonal_regional_full'
    for flag, value in [('--output-root',OUTPUT),('--core-block-sites','6000000'),('--flank-sites','0')]:
        result['command'][result['command'].index(flag)+1] = value
    result['expected'] = [OUTPUT+'/regional/'+suffix+'/COMPLETE.json']
    result['requires'] = list(dict.fromkeys(task['requires']+[gate]))
    result['replaces_task_id'] = task['id']
    return result


def build(original, source_manifest, orthogonal_source, gate=GATE):
    old, frozen = c.read(original), c.read(source_manifest)
    c.require(old['source'] == 'source-v2' and old['source_manifest_sha256'] == c.sha(source_manifest), 'Wrong frozen source identity')
    expected = frozen['files']['analysis/rerun/orthogonal.py']
    c.require(c.sha(orthogonal_source) == (expected['sha256'] if isinstance(expected,dict) else expected), 'Target-definition source hash mismatch')
    c.require(gate.startswith('{root}/validation/') and '..' not in gate and gate.endswith('.json'), 'Gate must be an explicit campaign validation marker')
    tree = ast.parse(Path(orthogonal_source).read_text())
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGETS' for t in n.targets)]
    c.require(len(values)==1, 'Missing frozen target inventory')
    targets = {(g,int(ch),p) for g,ch,pop,_ in values[0] for p in (pop,'YRI')}
    originals = [t for t in old['tasks'] if t['stage']=='orthogonal_regional']
    observed = {(c.flags(t)['--gene'],int(c.flags(t)['--chr']),c.flags(t)['--pop']) for t in originals}
    c.require(len(originals)==len(targets)==len(observed)==36 and observed==targets, 'Require exactly36 frozen regional targets')
    c.require(len({t['id'] for t in old['tasks']})==len(old['tasks']), 'Duplicate original task IDs')
    tasks = [transform(t,gate) for t in originals]
    return dict(schema=1,source='source-v2',source_manifest_sha256=old['source_manifest_sha256'],
        protocol='regional_full_context_v1',requires_gate=gate,validation_gate=gate,
        replaces=dict(original_manifest_sha256=c.sha(original),stage='orthogonal_regional',expected_targets=36),
        scientific_settings=dict(core_block_sites=6000000,flank_sites=0,pair_cap=20,pair_seed=42,
            mu=1.25e-8,rho=1e-8,context='whole population chromosome',window_half_bp=500000),tasks=tasks)


def bridge(old, fresh, old_npz, fresh_npz):
    """Prove unchanged observed panel while allowing corrected Gamma means."""
    oi, ni = old['identity'], fresh['identity']
    for key in oi['inputs']:
        c.require(key in ni['inputs'] and oi['inputs'][key]['sha256']==ni['inputs'][key]['sha256'], 'Regional input identity changed: '+key)
    c.require(set(oi['inputs'])==set(ni['inputs']), 'Regional input inventory changed')
    ignored={'output_root','core_block_sites','flank_sites'}
    c.require({k:v for k,v in oi['configuration'].items() if k not in ignored} ==
              {k:v for k,v in ni['configuration'].items() if k not in ignored}, 'Regional scientific configuration changed')
    c.require(ni['configuration']['core_block_sites']==6000000 and ni['configuration']['flank_sites']==0, 'Replacement is not full context')
    for key in ('cache','calibration','gamma_metadata','pair_seed','cxt_subset_seed','cxt_model','cxt_reps','cxt_device'):
        c.require(key in old['details'] and old['details'][key]==fresh['details'].get(key), 'Regional method/calibration changed: '+key)
    ready = fresh['details']['cache']['record']
    c.require(ready['shape'][1]<=6000000 and ready['members'].get('G.npy',{}).get('sha256'), 'Full context bound or genotype identity missing')
    with np.load(old_npz,allow_pickle=False) as a, np.load(fresh_npz,allow_pickle=False) as b:
        for key in ('gene','chromosome','population','gene_start','gene_end','window_start','window_end',
                    'pairs','population_hap_indices','window_positions','gamma_positions'):
            c.require(np.array_equal(a[key],b[key]), 'Regional observed data changed: '+key)
        c.require(b['pairs'].shape==(20,2) and b['gamma_mean'].shape==(len(b['gamma_positions']),20)
            and np.isfinite(b['gamma_mean']).all() and np.all(b['gamma_mean']>0), 'Invalid full-context Gamma output')
        c.require(b['cxt_log_tmrca'].shape==a['cxt_log_tmrca'].shape and np.isfinite(b['cxt_log_tmrca']).all(), 'Invalid replacement CXT output')
        return dict(observed_panel_identical=True,calibration_identical=True,
            gamma_changed=not np.array_equal(a['gamma_mean'],b['gamma_mean']),
            cxt_changed=not np.array_equal(a['cxt_log_tmrca'],b['cxt_log_tmrca']),
            original_cxt_provenance={k:v for k,v in old['details'].items() if k.startswith(('cxt_','torch_'))},
            selected_cxt_provenance={k:v for k,v in fresh['details'].items() if k.startswith(('cxt_','torch_'))},
            explanation='ASMC remains tied to its actual original regional marker. Both regional generations use identical immutable genotypes, positions, haplotypes, focal pairs and window; ASMC does not consume Gamma/CXT posterior values.')


def associate(root, original, asmc_manifest, full_manifest, output):
    audit, output = c.Audit(root), Path(output).resolve()
    c.require(not output.is_relative_to(audit.root) and not audit.root.is_relative_to(output), 'Output must be separate from synchronized host root')
    output.mkdir(parents=True,exist_ok=False)
    registry=dict(schema=1,complete=False,expected_associations=36,associations={},errors=[],
        collector_sha256=c.sha(__file__),host_root=str(audit.root),
        scope='One synchronized host: full-context Gamma and CXT selected together, associated with seeded ASMC retaining its actual older regional dependency.')
    try:
        old, full = c.read(original),c.read(full_manifest)
        original_hash, full_hash = audit.verify(original),audit.verify(full_manifest)
        expected = build(original,audit.path('source-v2/SOURCE_MANIFEST.json'),
                         audit.path('source-v2/analysis/rerun/orthogonal.py'),full['requires_gate'])
        c.require(full==expected, 'Full-context manifest changed beyond its explicit replacement protocol')
        hashes=audit.source('source-v2',full['source_manifest_sha256'])
        gate_path=audit.path(full['requires_gate'].format(root=str(audit.root)))
        gate_hash=audit.verify(gate_path);gate=c.read(gate_path)
        approved=gate.get('approved_task_manifest_sha256s',[])
        c.require(gate.get('passed') is True and gate.get('protocol')=='full_chromosome_v1'
                  and gate.get('source_manifest_sha256')==full['source_manifest_sha256'] and full_hash in approved,
                  'Full-context gate is not passed and explicitly bound to this source/task manifest')
        registry['full_context_gate_sha256']=gate_hash
        # Reuse complete strict ASMC validation, including all36 old dependency
        # products, native seeded gate, grids/subsets, receipts, and actual hashes.
        seeded=c.collect(root,original,asmc_manifest,output/'seeded_asmc')
        c.require(seeded['complete'], 'Seeded ASMC collection failed: '+str(seeded['errors']))
        registry['seeded_asmc_registry_sha256']=c.sha(output/'seeded_asmc/replacement_registry.json')
        old_tasks={t['id']:t for t in old['tasks']}
        for task in full['tasks']:
            f=c.flags(task);suffix=f['--gene']+'_'+f['--pop']
            old_task=old_tasks[task['replaces_task_id']]
            original_record, original_marker=audit.task(old_task,original_hash,old['source_manifest_sha256'],old['source'])
            full_record, full_marker=audit.task(task,full_hash,full['source_manifest_sha256'],full['source'])
            c.require(full_marker['identity']['inputs']['source']['sha256']==hashes['analysis/rerun/orthogonal.py'], 'Wrong regional source')
            checked=bridge(original_marker,full_marker,audit.path(original_record['results']),audit.path(full_record['results']))
            entry=seeded['replacements'][suffix]
            c.require(entry['regional_dependency']['complete_sha256']==original_record['complete_sha256'], 'ASMC refers to another original regional generation')
            registry['associations'][suffix]=dict(original_regional=original_record,full_regional=full_record,
                seeded_asmc=entry['corrected'],historical_asmc=entry['original'],panel_equivalence=checked)
        c.require(len(registry['associations'])==36, 'Incomplete regional association')
        for suffix,entry in registry['associations'].items():
            destination=output/'regional'/suffix;destination.mkdir(parents=True)
            for label in ('marker','results'):
                src=audit.path(entry['full_regional'][label]);dst=destination/src.name
                shutil.copyfile(src,dst);c.require(c.sha(src)==c.sha(dst),'Copy changed')
            entry['selected_regional_artifacts']={x.name:c.sha(x) for x in destination.iterdir()}
        registry['complete']=True
    except Exception as exc:
        registry['errors'].append(f'{type(exc).__name__}: {exc}')
    registry['verified_artifacts']=audit.artifacts
    c.write(output/'association_registry.json',registry)
    c.write(output/'validation.json',dict(passed=registry['complete'],expected_associations=36,
        validated_associations=len(registry['associations']),errors=registry['errors'],
        registry_sha256=c.sha(output/'association_registry.json')))
    return registry


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    m=s.add_parser('manifest');m.add_argument('--original-manifest',required=True)
    m.add_argument('--source-manifest',required=True);m.add_argument('--orthogonal-source',required=True)
    m.add_argument('--gate-marker',default=GATE);m.add_argument('--output',required=True)
    a=s.add_parser('associate');a.add_argument('--root',required=True)
    for flag in ('original-manifest','asmc-manifest','full-manifest','output-dir'):a.add_argument('--'+flag,required=True)
    x=p.parse_args()
    if x.action=='manifest':
        result=build(x.original_manifest,x.source_manifest,x.orthogonal_source,x.gate_marker)
        with Path(x.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        return 0
    result=associate(x.root,x.original_manifest,x.asmc_manifest,x.full_manifest,x.output_dir)
    print(json.dumps(dict(complete=result['complete'],errors=result['errors'])))
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
