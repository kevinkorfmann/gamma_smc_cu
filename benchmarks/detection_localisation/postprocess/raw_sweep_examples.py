"""Read-only raw profiles for the first evaluation replicate of each sweep history."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import tskit
ROOT=Path('/vast/projects/smathi/cohort/kkor/tmrca.cu/reruns/detection-localisation-20260927/production')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID')
out=ROOT/'raw_examples_v1';out.mkdir(exist_ok=False)
manifest=json.loads((ROOT/'manifest.json').read_text());norm=json.loads((ROOT/'summary_v1/normalisation.json').read_text())
records=[]
for regime in ['incomplete','near_fixed','completed','older_completed']:
    tasks=sorted([t for t in manifest['tasks'] if t['split']=='evaluation' and t['regime']==regime],key=lambda t:t['id'])
    task=tasks[0];sim=ROOT/'simulations'/task['id'];score=ROOT/'scores'/task['id']
    sr=json.loads((sim/'COMPLETE.json').read_text());qr=json.loads((score/'COMPLETE.json').read_text())
    assert sha(sim/'variants.trees')==sr['tree_sha256'] and sha(score/'raw_scores.npz')==qr['scores_sha256']
    ts=tskit.load(sim/'variants.trees');focal=[int(n) for n in ts.samples() if ts.node(int(n)).population==0];assert len(focal)==100
    q=np.load(score/'raw_scores.npz');target=int(q['selected_position0']);tree_rows=[];site_truth=np.full(len(q['gpos']),np.nan)
    checks=0
    for tree in ts.trees(tracked_samples=focal):
        weight=0;total=0.
        for node in tree.nodes():
            n=tree.num_tracked_samples(node)
            if n<2:continue
            pairs=n*(n-1)//2
            for child in tree.children(node):
                c=tree.num_tracked_samples(child);pairs-=c*(c-1)//2
            if pairs:total+=pairs*np.log(tree.time(node));weight+=pairs
        assert weight==4950
        avg=total/weight;lo,hi=np.searchsorted(q['gpos'],[tree.interval.left,tree.interval.right]);site_truth[lo:hi]=avg
        tree_rows.append((tree.interval.left-target,tree.interval.right-target,np.exp(avg)))
        if checks<5:
            direct=np.mean([np.log(tree.time(tree.mrca(x,y))) for i,x in enumerate(focal) for y in focal[i+1:]])
            assert np.isclose(avg,direct,rtol=1e-12,atol=1e-12);checks+=1
    bins=np.minimum((q['derived_frequency']*20).astype(int),19)
    ihs=np.abs((q['ihs']-np.array(norm['ihs_mean'])[bins])/np.array(norm['ihs_sd'])[bins])
    xp=(q['xp']-norm['xp_mean'])/norm['xp_sd']
    path=out/(regime+'.npz')
    np.savez_compressed(path,tree_profile=np.array(tree_rows),g_offset=q['gpos']-target,
        inferred_tmrca=np.exp(q['logtmrca']),true_tmrca_at_sites=np.exp(site_truth),
        i_offset=q['ipos']-target,ihs=ihs,x_offset=q['xpos']-target,xp=xp)
    centre=(q['gpos']>=target-10000)&(q['gpos']<target+10000)
    records.append(dict(regime=regime,id=task['id'],seed=task['seed'],selected_position0=target,
        source_tree_sha256=sr['tree_sha256'],source_scores_sha256=qr['scores_sha256'],
        direct_pair_checks=checks,output=path.name,output_sha256=sha(path),
        centre_true_tmrca=float(np.exp(site_truth[centre].mean())) if centre.sum()>=2 else None,
        centre_inferred_tmrca=float(np.exp(q['logtmrca'][centre].mean())) if centre.sum()>=2 else None))
receipt=dict(status='passed',selection='First replicate in each history by manifest ID, without visual selection',
    source_sha256=sha(__file__),slurm_job=os.environ['SLURM_JOB_ID'],records=records,
    outputs={r['output']:r['output_sha256'] for r in records})
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
