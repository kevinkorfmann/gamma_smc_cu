import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[1]/'regional_full_context.py'
spec=importlib.util.spec_from_file_location('regional_full',FILE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

@pytest.fixture
def manifest_inputs(tmp_path):
    root=tmp_path/'host';source=root/'source-v2/analysis/rerun/orthogonal.py'
    source.parent.mkdir(parents=True)
    targets=[(f'GENE{i}',1,'CEU','example') for i in range(18)]
    source.write_text('TARGETS='+repr(targets)+'\n')
    frozen=root/'source-v2/SOURCE_MANIFEST.json'
    write(frozen,dict(git_commit='a'*40,files={'analysis/rerun/orthogonal.py':m.c.sha(source)}))
    tasks=[]
    for gene,chrom,pop,_ in targets:
        for p in (pop,'YRI'):
            suffix=gene+'_'+p
            tasks.append(dict(id='orthogonal-regional-'+suffix,kind='gpu',stage='orthogonal_regional',
                command=['{python}','{source}/analysis/rerun/orthogonal.py','regional',
                    '--inputs-root','{inputs}','--tools-root','{tools}','--output-root','{root}/outputs/orthogonal',
                    '--chr','1','--gene',gene,'--pop',p,'--core-block-sites','65536','--flank-sites','8192',
                    '--pair-cap','20','--mu','1.25e-8','--rho','1e-8'],
                requires=['{inputs}/cache/parsed/chr1/READY.json'],
                expected=['{root}/outputs/orthogonal/regional/'+suffix+'/COMPLETE.json']))
    original=root/'control/tasks-orthogonal.json'
    write(original,dict(source='source-v2',source_manifest_sha256=m.c.sha(frozen),tasks=tasks))
    return root,original,frozen,source

def test_manifest_exact_changes_and_safe_independent_outputs(manifest_inputs):
    root,old,frozen,source=manifest_inputs
    before=old.read_bytes();result=m.build(old,frozen,source)
    assert len(result['tasks'])==36 and old.read_bytes()==before
    assert result['source_manifest_sha256']==m.c.sha(frozen)
    assert result['validation_gate']==result['requires_gate']==m.GATE
    assert len({x['expected'][0] for x in result['tasks']})==36
    for original,new in zip(m.c.read(old)['tasks'],result['tasks']):
        a,b=m.c.flags(original),m.c.flags(new)
        assert {k for k in a if a[k]!=b[k]}=={'--output-root','--core-block-sites','--flank-sites'}
        assert b['--core-block-sites']=='6000000' and b['--flank-sites']=='0'
        assert new['replaces_task_id']==original['id'] and m.GATE in new['requires']
        assert all('/outputs/orthogonal/' not in x for x in new['expected'])

@pytest.mark.parametrize('case',['duplicate','missing','wrong_core','wrong_source','foreign_target','bad_gate'])
def test_manifest_rejects_unbound_or_incomplete_inventory(manifest_inputs,case):
    root,old,frozen,source=manifest_inputs;data=m.c.read(old);gate=m.GATE
    if case=='duplicate':data['tasks'].append(copy.deepcopy(data['tasks'][0]))
    if case=='missing':data['tasks'].pop()
    if case=='wrong_core':data['tasks'][0]['command'][data['tasks'][0]['command'].index('--core-block-sites')+1]='999'
    if case=='wrong_source':data['source_manifest_sha256']='f'*64
    if case=='foreign_target':data['tasks'][0]['command'][data['tasks'][0]['command'].index('--gene')+1]='OTHER'
    if case=='bad_gate':gate='{root}/validation/../../unrelated.json'
    write(old,data)
    with pytest.raises(ValueError):m.build(old,frozen,source,gate)

@pytest.fixture
def bridge_inputs(tmp_path):
    ready=dict(shape=[204,1000],members={'G.npy':{'sha256':'a'*64}},positions_base=1)
    old=dict(identity=dict(configuration=dict(action='regional',gene='TEST',core_block_sites=65536,flank_sites=8192,output_root='/old',mu=1.25e-8),
        inputs={k:{'sha256':k,'path':'/same/'+k} for k in ('source','samples','genes','cache_manifest')}),
        details=dict(cache={'record':ready},calibration={'calibrated_Ne':10000},gamma_metadata={'time_units':'generations'},
            pair_seed=42,cxt_subset_seed=123,cxt_model='broad',cxt_reps=3,cxt_device='cpu',cxt_package='old',torch_version='old'))
    new=copy.deepcopy(old);new['identity']['configuration'].update(core_block_sites=6000000,flank_sites=0,output_root='/new')
    new['details'].update(cxt_package='new',torch_version='new')
    arrays=dict(gene='TEST',chromosome=1,population='CEU',gene_start=100,gene_end=200,window_start=-499850,window_end=500150,
        pairs=np.array([(0,i) for i in range(1,21)]),population_hap_indices=np.arange(102),window_positions=np.array([100,150,200]),
        gamma_positions=np.array([100,150,200]),gamma_mean=np.ones((3,20))*100,cxt_log_tmrca=np.ones((20,4)))
    a,b=tmp_path/'old.npz',tmp_path/'fresh.npz';np.savez(a,**arrays)
    arrays['gamma_mean']*=1.2;arrays['cxt_log_tmrca']*=1.01;np.savez(b,**arrays)
    return old,new,a,b

def test_bridge_preserves_old_identity_and_allows_changed_results(bridge_inputs):
    old,new,a,b=bridge_inputs;before=m.c.sha(a)
    result=m.bridge(old,new,a,b)
    assert result['observed_panel_identical'] and result['gamma_changed'] and result['cxt_changed']
    assert result['original_cxt_provenance']['cxt_package']=='old'
    assert result['selected_cxt_provenance']['cxt_package']=='new' and m.c.sha(a)==before

@pytest.mark.parametrize('case',['input','calibration','core','large_cache','pair','position','haps','nan_gamma','nan_cxt'])
def test_bridge_rejects_different_observations_or_invalid_products(bridge_inputs,case):
    old,new,a,b=bridge_inputs
    if case=='input':new['identity']['inputs']['cache_manifest']['sha256']='different'
    elif case=='calibration':new['details']['calibration']['calibrated_Ne']=9999
    elif case=='core':new['identity']['configuration']['core_block_sites']=65536
    elif case=='large_cache':
        old['details']['cache']['record']['shape'][1]=6000001
        new['details']['cache']['record']['shape'][1]=6000001
    else:
        with np.load(b) as z:arrays={k:z[k] for k in z.files}
        if case=='pair':arrays['pairs'][0,1]=99
        elif case=='position':arrays['window_positions'][0]=99
        elif case=='haps':arrays['population_hap_indices'][0]=999
        elif case=='nan_gamma':arrays['gamma_mean'][0,0]=np.nan
        elif case=='nan_cxt':arrays['cxt_log_tmrca'][0,0]=np.nan
        np.savez(b,**arrays)
    with pytest.raises(ValueError):m.bridge(old,new,a,b)

def test_association_fails_closed_without_explicit_manifest_bound_gate(manifest_inputs,tmp_path):
    root,old,frozen,source=manifest_inputs;full=root/'control/full.json'
    write(full,m.build(old,frozen,source))
    gate=root/'validation/regional-full-approved.json'
    write(gate,dict(passed=True,protocol='full_chromosome_v1',source_manifest_sha256=m.c.sha(frozen),approved_task_manifest_sha256s=['wrong']))
    result=m.associate(root,old,root/'control/asmc.json',full,tmp_path/'selected')
    assert not result['complete'] and 'explicitly bound' in result['errors'][0]
    assert not (tmp_path/'selected/regional').exists()
    assert not m.c.read(tmp_path/'selected/validation.json')['passed']


@pytest.mark.parametrize('passed,matching',[(False,True),(True,False),(True,True)])
def test_generated_custom_gate_is_validated_by_real_worker_before_dispatch(manifest_inputs,passed,matching):
    root,old,frozen,source=manifest_inputs
    full=root/'control/tasks-regional-full.json';definition=m.build(old,frozen,source);write(full,definition)
    write(root/'inputs/cache/parsed/chr1/READY.json',{'fixture':'external input'})
    write(root/'validation/regional-full-approved.json',dict(passed=passed,
        source_manifest_sha256=m.c.sha(frozen) if matching else 'f'*64))
    task=definition['tasks'][0]
    result=subprocess.run([sys.executable,str(FILE.with_name('worker.py')),'--root',str(root),
        '--manifest',str(full),'--kind','gpu','--only',task['id'],'--once','--idle-exit-seconds','0'],
        capture_output=True,text=True,timeout=15)
    attempt=root/'outputs/tasks'/task['id']/'attempt-001'
    if not passed or not matching:
        assert result.returncode!=0 and 'Production gate must pass' in result.stderr
        assert not attempt.exists() and not (root/'state'/f"{task['id']}.running.json").exists()
    else:
        # Fictional TARGETS-only script deliberately writes no scientific output;
        # reaching its failed receipt proves the correctly wired gate dispatched.
        assert result.returncode==0,result.stderr
        assert attempt.is_dir() and (root/'state'/f"{task['id']}.failed.json").is_file()
        assert not (root/'state'/f"{task['id']}.done.json").exists()


def test_all36_associate_with_seeded_products_without_rewriting_old_receipts(tmp_path):
    fixture_file=Path(__file__).with_name('test_collect_regional_asmc.py')
    spec=importlib.util.spec_from_file_location('regional_seeded_fixture',fixture_file)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    root,old_path,asmc_path=helper.campaign.__wrapped__(tmp_path)
    old=m.c.read(old_path)
    for t in old['tasks']:
        if t['stage']=='orthogonal_regional':t['command']+=['--core-block-sites','65536','--flank-sites','8192']
    write(old_path,old)
    am=m.c.read(asmc_path);am['replaces']['original_manifest_sha256']=m.c.sha(old_path);write(asmc_path,am)
    ready_path=root/'inputs/cache/parsed/chr1/READY.json';ready=m.c.read(ready_path)
    ready.update(shape=[204,3]);ready['members']['G.npy']={'sha256':'a'*64};write(ready_path,ready)
    full_path=root/'control/full.json'
    full=m.build(old_path,root/'source-v2/SOURCE_MANIFEST.json',root/'source-v2/analysis/rerun/orthogonal.py');write(full_path,full)
    full_tasks={t['replaces_task_id']:t for t in full['tasks']}
    for task in old['tasks']:
        if task['stage']!='orthogonal_regional':continue
        suffix=task['id'].removeprefix('orthogonal-regional-')
        folder=root/'outputs/orthogonal/regional'/suffix;marker=m.c.read(folder/'COMPLETE.json')
        marker['identity']['inputs']['cache_manifest']=helper.fingerprint(root,ready_path)
        marker['details'].update(cache={'record':ready},calibration={'calibrated_Ne':10000},
            gamma_metadata={'time_units':'generations'},pair_seed=42,cxt_subset_seed=123,cxt_model='broad',
            cxt_reps=3,cxt_device='cpu',cxt_package='fictional',torch_version='fictional')
        helper.finish_task(root,task,'source-v2',m.c.sha(old_path),old['source_manifest_sha256'],folder,
            marker['identity']['inputs'],marker['details'])
        marker=m.c.read(folder/'COMPLETE.json');marker['identity']['configuration'].update(core_block_sites=65536,flank_sites=8192)
        write(folder/'COMPLETE.json',marker);helper.rehash_product(root,task['id'],folder)
        fresh=root/'outputs/orthogonal-full/regional'/suffix;fresh.mkdir(parents=True)
        with np.load(folder/'results.npz') as z:arrays={k:z[k] for k in z.files}
        arrays['gamma_mean']*=1.1;arrays['cxt_log_tmrca']*=1.05;np.savez(fresh/'results.npz',**arrays)
        new_task=full_tasks[task['id']]
        helper.finish_task(root,new_task,'source-v2',m.c.sha(full_path),old['source_manifest_sha256'],fresh,
            marker['identity']['inputs'],marker['details'])
        n=m.c.read(fresh/'COMPLETE.json');n['identity']['configuration'].update(core_block_sites=6000000,flank_sites=0)
        write(fresh/'COMPLETE.json',n);helper.rehash_product(root,new_task['id'],fresh)
        for historical in (False,True):
            scope='source-v2' if historical else 'source-runtime-v3'
            af=root/('outputs/orthogonal/asmc' if historical else 'outputs/orthogonal-asmc-v3/asmc')/suffix
            if not af.exists():continue
            aid=('orthogonal-asmc-' if historical else 'orthogonal-v3-asmc-')+suffix
            definition=next(x for x in (old['tasks'] if historical else am['tasks']) if x['id']==aid)
            ac=m.c.read(af/'COMPLETE.json');ac['identity']['inputs']['cache_manifest']=helper.fingerprint(root,ready_path)
            ac['identity']['inputs']['regional_marker']=helper.fingerprint(root,folder/'COMPLETE.json')
            ac['details'].update(cache={'record':ready},regional_dependency=helper.fingerprint(root,folder/'COMPLETE.json'))
            helper.finish_task(root,definition,scope,m.c.sha(old_path if historical else asmc_path),
                old['source_manifest_sha256'] if historical else am['source_manifest_sha256'],af,ac['identity']['inputs'],ac['details'])
    write(root/'validation/regional-full-approved.json',dict(passed=True,protocol='full_chromosome_v1',
        source_manifest_sha256=old['source_manifest_sha256'],approved_task_manifest_sha256s=[m.c.sha(full_path)]))
    retained=root/'outputs/orthogonal/regional/GENE0_CEU/COMPLETE.json';before=m.c.sha(retained)
    result=m.associate(root,old_path,asmc_path,full_path,tmp_path/'selected')
    assert result['complete'],result['errors']
    assert len(result['associations'])==36 and m.c.sha(retained)==before
    entry=result['associations']['GENE0_CEU']
    assert entry['original_regional']['complete_sha256']==before
    assert entry['full_regional']['task_id']=='orthogonal-full-regional-GENE0_CEU'
    assert entry['seeded_asmc']['task_id']=='orthogonal-v3-asmc-GENE0_CEU'
    assert entry['panel_equivalence']['gamma_changed'] and entry['panel_equivalence']['cxt_changed']
    assert np.load(tmp_path/'selected/regional/GENE0_CEU/results.npz')['gamma_mean'][0,0]==pytest.approx(110)
    assert np.load(tmp_path/'selected/seeded_asmc/asmc/GENE0_CEU/results.npz')['mean'][0,0]==100
