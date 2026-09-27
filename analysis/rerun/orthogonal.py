#!/usr/bin/env python3
"""Portable, fail-fast orthogonal-analysis tasks; see ORTHOGONAL.md."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
import time
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'analysis/genome_wide'))
from rerun_support import load_input, fingerprint, calibrate
from build_candidates import POPS, SUPERPOPS
from infer_chromosome import load_samples, get_population_haplotype_indices

TARGETS=[('GRK2',11,'GIH','case'),('TREM2',6,'IBS','case'),('IFIH1',2,'IBS','case'),
 ('BPIFA2',20,'GIH','case'),('SLC6A15',12,'CHS','case'),('CCDC92',12,'CDX','case'),('CLEC6A',12,'CDX','case'),
 ('SLC24A5',15,'GBR','positive'),('LCT',2,'CEU','positive'),('EDAR',2,'CHB','positive'),('ABCC11',16,'CHB','positive'),('KITLG',12,'MXL','positive'),
 ('ADAM22',7,'CDX','control'),('CCDC70',13,'GIH','control'),('TMEM30A',6,'CDX','control'),('ZNF420',19,'CDX','control'),('RAB11FIP3',16,'GIH','control'),('C11orf65',11,'GIH','control')]
ACTIONS=['manifest','selscan','normalize','h12','h12-aggregate','neighborhood-background','neighborhood-aggregate','variant','neighborhood','xpehh','regional','asmc']


def legacy(name):
    path=REPO/'analysis/orthogonal_v41/scripts'/name
    spec=importlib.util.spec_from_file_location('rerun_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=ACTIONS)
    p.add_argument('--inputs-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--tools-root',required=True)
    p.add_argument('--regional-root',help='ASMC only: root containing completed regional tasks (defaults to --output-root)')
    p.add_argument('--cache-dir');p.add_argument('--samples');p.add_argument('--genes-dir')
    p.add_argument('--chr',type=int);p.add_argument('--pop',choices=POPS);p.add_argument('--gene')
    p.add_argument('--targets',help='Optional TSV gene,chr,pop,group; overrides the18 explicit manuscript targets')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--selscan-bin');p.add_argument('--decoding-quantities')
    p.add_argument('--window-sites',type=int,default=400);p.add_argument('--step-sites',type=int,default=50)
    p.add_argument('--half-window-bp',type=int,default=500000);p.add_argument('--integration-pad-bp',type=int,default=2000000)
    p.add_argument('--mu',type=float,default=1.25e-8);p.add_argument('--rho',type=float,default=1e-8)
    p.add_argument('--core-block-sites',type=int,default=65536);p.add_argument('--flank-sites',type=int,default=8192)
    p.add_argument('--pair-cap',type=int,default=20);p.add_argument('--resume',action='store_true')
    return p


def configure(a):
    base=Path(a.inputs_root).resolve();a.cache_dir=str(Path(a.cache_dir or base/'cache').resolve())
    a.samples=str(Path(a.samples or base/'samples.txt').resolve());a.genes_dir=str(Path(a.genes_dir or Path(a.cache_dir)/'genes').resolve())
    output=Path(a.output_root).resolve()
    if a.regional_root:a.regional_root=str(Path(a.regional_root).resolve())
    if output==base or output in base.parents or base in output.parents:raise ValueError('Inputs and outputs must be separate non-nested roots')
    if a.action!='manifest' and (not Path(a.samples).is_file() or not Path(a.genes_dir).is_dir()):raise FileNotFoundError('Samples and gene annotations are required')
    if a.action not in ['manifest','normalize','h12-aggregate','neighborhood-aggregate'] and (a.chr not in range(1,23) or a.pop is None):raise ValueError('This task requires --chr1..22 and --pop')
    if a.action in ['normalize','h12-aggregate','neighborhood-aggregate'] and a.pop is None:raise ValueError('Population is required')
    if a.action in ['variant','neighborhood','xpehh','regional','asmc'] and not a.gene:raise ValueError('Gene is required')
    if a.mu<=0 or a.rho<0:raise ValueError('Invalid physicalrates')
    if min(a.threads,a.window_sites,a.step_sites,a.half_window_bp,a.pair_cap,a.core_block_sites)<=0 or a.flank_sites<0 or a.integration_pad_bp<0:raise ValueError('Invalid numerical settings')
    return a


def target_table(a):
    t=pd.read_csv(a.targets,sep='\t') if a.targets else pd.DataFrame(TARGETS,columns=['gene','chr','pop','group'])
    if not {'gene','chr','pop','group'}.issubset(t):raise ValueError('Target TSV must have gene,chr,pop,group')
    if t[['gene','pop']].duplicated().any():raise ValueError('Duplicate gene-population target')
    if not set(t['pop']).issubset(POPS):raise ValueError('Unknown target population')
    if not t['chr'].isin(range(1,23)).all():raise ValueError('Target chromosomes must be integers1..22')
    if t[['gene','chr','pop','group']].isna().any().any():raise ValueError('Target annotations cannot be missing')
    return t


def output_folder(a):
    if a.action=='manifest':return Path(a.output_root)/'manifest'
    elif a.action in ['normalize','h12-aggregate','neighborhood-aggregate']:key=a.pop
    elif a.gene:key=f'{a.gene}_{a.pop}'
    else:key=f'chr{a.chr}_{a.pop}'
    return Path(a.output_root)/a.action/key


def identity(a):
    d={k:v for k,v in vars(a).items() if k!='resume'}
    files={'source':fingerprint(__file__)}
    for name in ['rerun_support.py','build_candidates.py','infer_chromosome.py']:
        files[name]=fingerprint(REPO/'analysis/genome_wide'/name)
    files['cache.py']=fingerprint(REPO/'analysis/rerun/cache.py')
    if a.targets:files['targets']=fingerprint(a.targets)
    if a.action!='manifest':files['samples']=fingerprint(a.samples)
    if a.action=='selscan':files['selscan_binary']=fingerprint(a.selscan_bin or Path(a.tools_root)/'selscan/bin/linux/selscan')
    if a.action=='asmc':files['decoding_quantities']=fingerprint(a.decoding_quantities or Path(a.tools_root)/'asmc_data/CEU_csfs50/decoding.decodingQuantities.gz')
    if a.action=='normalize':files['all_gene_annotations']=[fingerprint(Path(a.genes_dir)/f'chr{chrom}_genes.tsv') for chrom in range(1,23)]
    if a.chr:files['genes']=fingerprint(Path(a.genes_dir)/f'chr{a.chr}_genes.tsv')
    if a.action!='manifest':
        for name in ['run_selscan_task.py','run_three_method.py','run_asmc_region.py']:
            files[name]=fingerprint(REPO/'analysis/orthogonal_v41/scripts'/name)
        if a.chr:
            cache=Path(a.cache_dir);cache=cache/'parsed' if (cache/'parsed').is_dir() else cache
            ready=cache/f'chr{a.chr}'/'READY.json'
            files['cache_manifest']=fingerprint(ready if ready.exists() else cache/f'chr{a.chr}.npz')
        deps={'normalize':'selscan','h12-aggregate':'h12','neighborhood-aggregate':'neighborhood-background'}
        if a.action in deps:
            files['dependency_markers']=[fingerprint(Path(a.output_root)/deps[a.action]/f'chr{chrom}_{a.pop}'/'COMPLETE.json') for chrom in range(1,23)]
        if a.action=='asmc':files['regional_marker']=fingerprint(Path(a.regional_root or a.output_root)/'regional'/f'{a.gene}_{a.pop}'/'COMPLETE.json')
    return {'configuration':d,'inputs':files}


def begin(a):
    folder=output_folder(a);expected=identity(a);marker=folder/'COMPLETE.json'
    if folder.exists():
        if a.resume and marker.exists():
            old=json.loads(marker.read_text())
            if old.get('identity')==expected and all((folder/name).is_file() and fingerprint(folder/name)['sha256']==sha for name,sha in old['outputs'].items()):return folder,expected,True
        raise FileExistsError(f'Refusing to replace existing/incomplete task directory {folder}')
    folder.mkdir(parents=True);return folder,expected,False


def finish(folder,ident,extra):
    outputs={str(p.relative_to(folder)):fingerprint(p)['sha256'] for p in folder.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    record={'complete':True,'identity':ident,'outputs':outputs,'details':extra}
    temporary=folder/'COMPLETE.json.tmp'
    temporary.write_text(json.dumps(record,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    temporary.replace(folder/'COMPLETE.json')


def require_complete(folder):
    marker=Path(folder)/'COMPLETE.json'
    if not marker.is_file():raise FileNotFoundError(f'Missing completed dependency {folder}')
    record=json.loads(marker.read_text())
    if not record.get('complete'):raise ValueError(f'Incomplete dependency {folder}')
    for name,sha in record['outputs'].items():
        path=Path(folder)/name
        if not path.is_file() or fingerprint(path)['sha256']!=sha:raise ValueError(f'Dependency checksum mismatch {path}')
    return record


def chromosome(a):
    data,source=load_input(a.cache_dir,a.chr);pop_map=load_samples(a.samples)
    if any(str(s) not in pop_map for s in data['sample_ids']):raise ValueError('Sample membership missing for cache samples')
    haps=np.array(get_population_haplotype_indices(data['sample_ids'],pop_map,a.pop),dtype=np.int64)
    if len(haps)<4:raise ValueError('Fewer than4 focal haplotypes')
    return data,pop_map,haps,source


def gene(a):
    genes=pd.read_csv(Path(a.genes_dir)/f'chr{a.chr}_genes.tsv',sep='\t')
    g=genes[genes.gene_name==a.gene]
    if len(g)!=1:raise ValueError(f'Expected exactly one gene annotation for {a.gene}, got{len(g)}')
    g=g.iloc[0];mid=(int(g.start)+int(g.end))//2
    return int(g.start),int(g.end),mid,mid-a.half_window_bp,mid+a.half_window_bp


def window_slice(G,positions,haps,lo,hi):
    ix=np.arange(np.searchsorted(positions,lo,side='left'),np.searchsorted(positions,hi,side='right'))
    return np.ascontiguousarray(G[np.ix_(np.asarray(haps,dtype=int),ix)]),np.asarray(positions[ix])


def h12_track(G,positions,width=400,step=50):
    mids=[];h12=[];h2h1=[]
    for s in range(0,G.shape[1]-width+1,step):
        counts=np.array(list(Counter(row.tobytes() for row in G[:,s:s+width]).values()))
        f=np.sort(counts.astype(float)/len(G))[::-1];h1=np.sum(f*f)
        h12.append(float((f[:2].sum())**2+np.sum(f[2:]**2)))
        h2h1.append(float(np.sum(f[1:]**2)/h1)) # zero, not1, for one unique haplotype
        mids.append(int(positions[s+width//2]))
    return np.array(mids),np.array(h12),np.array(h2h1)


def task_table(a):
    """Return the canonical task inventory without performing filesystem work."""
    t=target_table(a);rows=[]
    for chrom in range(1,23):
        for pop in POPS:
            for action in ['selscan','h12']:rows.append({'action':action,'chr':chrom,'pop':pop,'gene':''})
    for pop in POPS:
        for action in ['normalize','h12-aggregate']:rows.append({'action':action,'chr':'','pop':pop,'gene':''})
    for pop in ['GIH','CDX','CHS']:
        for chrom in range(1,23):rows.append({'action':'neighborhood-background','chr':chrom,'pop':pop,'gene':''})
        rows.append({'action':'neighborhood-aggregate','chr':'','pop':pop,'gene':''})
    for g in t.itertuples():
        for action in ['variant','neighborhood','xpehh']:rows.append({'action':action,'chr':g.chr,'pop':g.pop,'gene':g.gene})
        for pop in sorted({g.pop,'YRI'}):
            for action in ['regional','asmc']:rows.append({'action':action,'chr':g.chr,'pop':pop,'gene':g.gene})
    for pop in POPS:rows.append({'action':'neighborhood','chr':6,'pop':pop,'gene':'TREM2'})
    for g in t.itertuples():rows.append({'action':'neighborhood','chr':g.chr,'pop':'YRI','gene':g.gene})
    tasks=pd.DataFrame(rows).drop_duplicates();tasks.insert(0,'task_id',range(len(tasks)))
    return tasks,t


def manifest(a,folder):
    tasks,t=task_table(a)
    tasks.to_csv(folder/'tasks.tsv',sep='\t',index=False);t.to_csv(folder/'targets.tsv',sep='\t',index=False)
    return {'n_tasks':len(tasks),'task_counts':tasks.action.value_counts().to_dict(),'dependencies':{'normalize':'all22 selscan chromosomes for its population','h12-aggregate':'all22 h12 chromosomes for its population','asmc':'matching gene/pop regional task','neighborhood-aggregate':'all22 neighborhood-background chromosomes for its population'},'note':'Controls are retained manuscript controls; their neutral status must be reevaluated against rerun ranks.'}


def selscan(a,folder):
    executable=Path(a.selscan_bin or Path(a.tools_root)/'selscan/bin/linux/selscan')
    if not executable.is_file() or not os.access(executable,os.X_OK):raise FileNotFoundError(f'selscan executable unavailable: {executable}')
    data,pop_map,haps,source=chromosome(a);G=np.ascontiguousarray(data['G'][haps,:]);pos=data['positions']
    n=G.sum(axis=0);poly=(n>0)&(n<len(G));G=G[:,poly];pos=np.asarray(pos)[poly]
    if G.shape[1]<2:raise ValueError('No polymorphic sites for selscan')
    helper=legacy('run_selscan_task.py');hap,mapfile=helper.write_hap_map(str(folder/'data'),a.chr,pos,G)
    commands=[]
    for stat in ['ihs','nsl']:
        command=[str(executable),'--'+stat,'--hap',hap,'--map',mapfile,'--out',str(folder/stat),'--threads',str(a.threads),'--maf','0.05']
        r=subprocess.run(command,capture_output=True,text=True)
        (folder/(stat+'.stdout.log')).write_text(r.stdout);(folder/(stat+'.stderr.log')).write_text(r.stderr)
        if r.returncode:raise RuntimeError(f'{stat} failed with return code {r.returncode}; see {folder}')
        if not (folder/f'{stat}.{stat}.out').is_file():raise FileNotFoundError(f'Missing {stat} result')
        commands.append(command)
    Path(hap).unlink();Path(mapfile).unlink()
    return {'cache':source,'commands':commands,'executable':fingerprint(executable),'haplotypes':len(G),'polymorphic_sites':len(pos),'map':'uniform1cM/Mb','alleles':'VCF ALT/REF coding; no ancestral-state polarization'}


def load_selscan(path,stat):
    # Selscan releases emit six columns (ID POS FREQ IHH1 IHH0 SCORE), or
    # prepend CHR. Detect exact field count and coerce headers explicitly.
    with open(path) as f:first=f.readline().strip().split()
    n=len(first)
    if n not in (6,7):raise ValueError(f'Unexpected selscan column count {n}: {path}')
    names=['id','pos','freq','ihh1','ihh0',stat] if n==6 else ['chr','id','pos','freq','ihh1','ihh0',stat]
    d=pd.read_csv(path,sep=r'\s+',header=None,names=names)
    for c in ['pos','freq',stat]:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['pos','freq',stat]);d=d[np.isfinite(d[stat]) & d.freq.between(0,1)].copy()
    d['pos']=d.pos.astype(np.int64)
    return d


def normalize(a,folder):
    summaries={};dependencies=[];normalized={}
    for stat in ['ihs','nsl']:
        frames=[]
        for chrom in range(1,23):
            root=Path(a.output_root)/'selscan'/f'chr{chrom}_{a.pop}';marker=require_complete(root)
            path=root/f'{stat}.{stat}.out';d=load_selscan(path,stat);d['chromosome']=chrom;frames.append(d)
            dependencies.append(fingerprint(root/'COMPLETE.json'))
        d=pd.concat(frames,ignore_index=True);d['frequency_bin']=pd.cut(d.freq,np.linspace(0,1,21),include_lowest=True,labels=False)
        d['normalized']=np.nan;bins=[]
        for b in range(20):
            mask=d.frequency_bin==b;vals=d.loc[mask,stat];sd=float(vals.std(ddof=0));mean=float(vals.mean())
            usable=len(vals)>=50 and np.isfinite(sd) and sd>0
            if usable:d.loc[mask,'normalized']=(vals-mean)/sd
            bins.append({'bin':b,'n':len(vals),'mean':mean,'sd':sd,'usable':usable})
        normalized[stat]=d;pd.DataFrame(bins).to_csv(folder/(stat+'_normalization_bins.csv'),index=False)
        d.to_csv(folder/(stat+'_normalized.tsv.gz'),sep='\t',index=False)
    rows=[]
    for chrom in range(1,23):
        genes=pd.read_csv(Path(a.genes_dir)/f'chr{chrom}_genes.tsv',sep='\t')
        tracks={stat:d[d.chromosome==chrom].sort_values('pos') for stat,d in normalized.items()}
        for g in genes.itertuples():
            row={'chr':chrom,'gene_name':g.gene_name,'gstart':g.start,'gend':g.end}
            for stat,d in tracks.items():
                left=np.searchsorted(d.pos.to_numpy(),g.start,side='left');right=np.searchsorted(d.pos.to_numpy(),g.end,side='right')
                vals=d.normalized.iloc[left:right].dropna()
                row['n_'+stat+'_sites']=len(vals);row['max_abs_'+stat+'_norm']=float(vals.abs().max()) if len(vals) else np.nan
                row['frac_'+stat+'_extreme']=float((vals.abs()>2).mean()) if len(vals) else np.nan
            rows.append(row)
    table=pd.DataFrame(rows)
    for stat in ['ihs','nsl']:
        for col in ['max_abs_'+stat+'_norm','frac_'+stat+'_extreme']:table[col+'_rank']=table[col].rank(ascending=False,pct=True,na_option='keep')
    table.to_csv(folder/'genelevel.csv',index=False)
    return {'dependencies':dependencies,'normalization':'population-wide all22 autosomes;20 ALT-frequency bins;ddof0;minimum50 finite sites/bin','undefined_scores':'NaN ranks retained; no rank is assigned to an absent computable score','genes':len(table)}


def h12(a,folder):
    data,pm,haps,source=chromosome(a);G=np.ascontiguousarray(data['G'][haps,:]);pos=data['positions']
    sums=G.sum(axis=0);poly=(sums>0)&(sums<len(G));G=G[:,poly];pos=np.asarray(pos)[poly]
    mids,values,h2=h12_track(G,pos,a.window_sites,a.step_sites)
    np.savez_compressed(folder/'windows.npz',midpoint=mids,H12=values,H2_H1=h2)
    genes=pd.read_csv(Path(a.genes_dir)/f'chr{a.chr}_genes.tsv',sep='\t');rows=[]
    for g in genes.itertuples():
        ix=np.flatnonzero((mids>=g.start)&(mids<=g.end));peak=ix[np.argmax(values[ix])] if len(ix) else None
        rows.append({'chr':a.chr,'gene_name':g.gene_name,'gstart':g.start,'gend':g.end,'n_windows':len(ix),'max_h12':float(values[peak]) if peak is not None else np.nan,'h2h1_at_peak':float(h2[peak]) if peak is not None else np.nan})
    pd.DataFrame(rows).to_csv(folder/'genes.csv',index=False)
    return {'cache':source,'site_convention':'remove focal monomorphic sites','n_windows':len(mids),'midpoint':'central retained SNP position','width':a.window_sites,'step':a.step_sites}


def h12_aggregate(a,folder):
    frames=[];deps=[]
    for chrom in range(1,23):
        root=Path(a.output_root)/'h12'/f'chr{chrom}_{a.pop}';require_complete(root);deps.append(fingerprint(root/'COMPLETE.json'));frames.append(pd.read_csv(root/'genes.csv'))
    d=pd.concat(frames,ignore_index=True);d['h12_percentile']=d.max_h12.rank(method='max',pct=True)*100;d.to_csv(folder/'genelevel.csv',index=False)
    return {'dependencies':deps,'n_finite':int(d.max_h12.notna().sum()),'percentiles':'weakempiricalCDF among finite gene summaries'}



def neighborhood_background(a,folder):
    data,pm,haps,source=chromosome(a)
    genes=pd.read_csv(Path(a.genes_dir)/f'chr{a.chr}_genes.tsv',sep='\t');rows=[]
    # Windows restart at each gene's +/-500kb boundary, as in the historical
    # SI comparison; a whole-chromosome track has a different phase.
    for g in genes.itertuples():
        mid=(int(g.start)+int(g.end))//2;lo=mid-a.half_window_bp;hi=mid+a.half_window_bp
        G,pos=window_slice(data['G'],data['positions'],haps,lo,hi)
        mids,values,h2=h12_track(G,pos,a.window_sites,a.step_sites)
        peak=int(np.argmax(values)) if len(values) else None
        rows.append({'chr':a.chr,'gene_name':g.gene_name,'gstart':g.start,'gend':g.end,'n_windows':len(values),'max_h12':float(values[peak]) if peak is not None else np.nan,'h2h1_at_peak':float(h2[peak]) if peak is not None else np.nan})
    pd.DataFrame(rows).to_csv(folder/'genes.csv',index=False)
    return {'cache':source,'site_convention':'allcohortVCFsites;focalmonomorphicretained','window':'midpoint+-half-window-bp;windowsrestartforeachgene','width':a.window_sites,'step':a.step_sites}


def neighborhood_aggregate(a,folder):
    frames=[];deps=[]
    for chrom in range(1,23):
        root=Path(a.output_root)/'neighborhood-background'/f'chr{chrom}_{a.pop}';require_complete(root);deps.append(fingerprint(root/'COMPLETE.json'));frames.append(pd.read_csv(root/'genes.csv'))
    d=pd.concat(frames,ignore_index=True)
    d['h12_percentile']=d.max_h12.rank(method='max',pct=True)*100
    d['h2h1_percentile']=d.h2h1_at_peak.rank(method='max',pct=True)*100
    d.to_csv(folder/'genelevel.csv',index=False)
    return {'dependencies':deps,'finite_summaries':int(d.max_h12.notna().sum()),'percentiles':'weakempiricalCDF amongfinitegenes'}

def variant(a,folder):
    data,pm,haps,source=chromosome(a);gs,ge,mid,lo,hi=gene(a);sp=next(s for s,pp in SUPERPOPS.items() if a.pop in pp)
    other=[h for i,s in enumerate(data['sample_ids']) if pm[str(s)][1]!=sp for h in [2*i,2*i+1]]
    f,pos=window_slice(data['G'],data['positions'],haps,lo,hi);o,_=window_slice(data['G'],data['positions'],other,lo,hi)
    pf=f.mean(axis=0);po=o.mean(axis=0);keep=~((pf==0)&(po==0))&~((pf==1)&(po==1));pf,po,pos=pf[keep],po[keep],pos[keep]
    if not len(pos):raise ValueError('No comparison-polymorphic variant')
    num=(pf-po)**2-pf*(1-pf)/(len(f)-1)-po*(1-po)/(len(o)-1);den=pf*(1-po)+po*(1-pf)
    fst=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);diff=pf-po;ext=int(np.argmax(abs(diff)));D=int((diff<0).sum());E=int((diff>0).sum())
    sites=pd.DataFrame({'position':pos,'focal_ALT_AF':pf,'other_ALT_AF':po,'delta_AF':diff,'hudson_FST':fst})
    for population in POPS:
        ph=get_population_haplotype_indices(data['sample_ids'],pm,population)
        pg,_=window_slice(data['G'],data['positions'],ph,lo,hi)
        sites[population+'_ALT_AF']=pg.mean(axis=0)[keep]
    sites.to_csv(folder/'variants.tsv.gz',sep='\t',index=False)
    row={'gene':a.gene,'chr':a.chr,'pop':a.pop,'superpopulation':sp,'gene_start':gs,'gene_end':ge,'window_start':lo,'window_end':hi,'n_variants':len(pos),'n_depleted':D,'n_enriched':E,'depleted_to_enriched_ratio':D/E if E else None,'max_fst':float(np.nanmax(fst)),'mean_top10_fst':float(np.sort(fst[np.isfinite(fst)])[-10:].mean()),'most_diff_position':int(pos[ext]),'most_diff_focal_AF':float(pf[ext]),'most_diff_other_AF':float(po[ext]),'most_diff_delta_AF':float(diff[ext])}
    (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n');return {'cache':source,'window':'gene midpoint +- half-window-bp','comparison':'focal pop vs pooled nonfocal-superpopulation haplotypes','summary':row}


def neighborhood(a,folder):
    data,pm,haps,source=chromosome(a);gs,ge,mid,lo,hi=gene(a);G,pos=window_slice(data['G'],data['positions'],haps,lo,hi)
    mids,h12,h2=h12_track(G,pos,a.window_sites,a.step_sites)
    if not len(h12):raise ValueError('Insufficient cohort sites for neighborhood H12')
    np.savez_compressed(folder/'windows.npz',midpoint=mids,H12=h12,H2_H1=h2)
    ix=int(np.argmax(h12));return {'cache':source,'site_convention':'allcohortVCFsites, including focalmonomorphic','window':'midpoint +- half-window-bp','max_H12':float(h12[ix]),'H2_H1_at_peak':float(h2[ix]),'peak_position':int(mids[ix])}


def xpehh(a,folder):
    import allel
    data,pm,haps,source=chromosome(a);gs,ge,mid,lo,hi=gene(a);sp=next(s for s,pp in SUPERPOPS.items() if a.pop in pp)
    focal=[h for i,s in enumerate(data['sample_ids']) if pm[str(s)][1]==sp for h in [2*i,2*i+1]]
    reference=get_population_haplotype_indices(data['sample_ids'],pm,'YRI')
    if set(focal)&set(reference):raise ValueError('Focal and reference XP-EHH haplotypes overlap; specify a scientifically appropriate contrast outside this fixed YRI comparison')
    lower=gs-a.half_window_bp;upper=ge+a.half_window_bp
    f,pos=window_slice(data['G'],data['positions'],focal,lower-a.integration_pad_bp,upper+a.integration_pad_bp)
    r,_=window_slice(data['G'],data['positions'],reference,lower-a.integration_pad_bp,upper+a.integration_pad_bp)
    h1=allel.HaplotypeArray(f.T);h2=allel.HaplotypeArray(r.T);c1=h1.count_alleles();c2=h2.count_alleles()
    # Preserve historical filter; it excludes populations fixed for ALT.
    keep=(c1.is_segregating()|c2.is_segregating())&c1.is_biallelic()&c2.is_biallelic()
    pos=pos[keep]
    if len(pos)<2:raise ValueError('Insufficient XP-EHH variants after specified biallelic filter')
    xp=allel.xpehh(h1[keep],h2[keep],pos,use_threads=False);mask=(pos>=lower)&(pos<=upper)&np.isfinite(xp)
    if not mask.any():raise ValueError('No finite XP-EHH score inside reporting region')
    pd.DataFrame({'position':pos,'XP_EHH_raw':xp}).to_csv(folder/'sites.tsv.gz',sep='\t',index=False)
    return {'cache':source,'superpopulation':sp,'reference':'YRI','mean_raw_XP_EHH':float(xp[mask].mean()),'max_raw_XP_EHH':float(xp[mask].max()),'n_finite':int(mask.sum()),'window':'gene body plus500kb;additional2Mb integration padding','normalization':'none;rawscikit-allel values'}


def regional(a,folder):
    # Preserve cxt/torch-first loading required by the historical environment.
    import torch
    import cxt
    import gamma_smc_cu
    data,pm,haps,source=chromosome(a);gs,ge,mid,lo,hi=gene(a)
    full=np.ascontiguousarray(data['G'][haps,:]);pos=data['positions'];calibration=calibrate(full,pos,a.mu,a.rho)
    all_pairs=np.array([(i,j) for i in range(len(haps)) for j in range(i+1,len(haps))]);rng=np.random.default_rng(42)
    ix=np.sort(rng.choice(len(all_pairs),min(a.pair_cap,len(all_pairs)),replace=False));pairs=all_pairs[ix].tolist()
    result=gamma_smc_cu.infer_blockwise(full,pos,pairs=pairs,mu=a.mu,rho=a.rho,Ne=calibration['calibrated_Ne'],physical_mu=a.mu,auto_estimate_theta=False,mean_only=True,core_block_sites=a.core_block_sites,flank_sites=a.flank_sites)
    selected=(result['positions']>=lo)&(result['positions']<=hi)
    Gwin,pwin=window_slice(full,pos,np.arange(len(haps)),lo,hi)
    if len(haps)<50:raise ValueError('cxt broad requires50 haplotypes')
    helper=legacy('run_three_method.py');cx=helper.run_cxt(Gwin,pwin,pairs,lo,hi)
    if 'error' in cx:raise RuntimeError(cx['error'])
    if not np.isfinite(cx['log_tmrca']).all():raise ValueError('Nonfinite cxt output')
    np.savez_compressed(folder/'results.npz',gene=a.gene,chromosome=a.chr,population=a.pop,gene_start=gs,gene_end=ge,window_start=lo,window_end=hi,pairs=np.asarray(pairs),population_hap_indices=haps,gamma_positions=result['positions'][selected],gamma_mean=result['mean'][selected,:],cxt_log_tmrca=cx['log_tmrca'],window_positions=pwin)
    return {'cache':source,'calibration':calibration,'gamma_metadata':result.get('metadata',{}),'pair_seed':42,'cxt_subset_seed':123,'cxt_model':'broad','cxt_reps':3,'cxt_device':'cpu','cxt_package':getattr(cxt,'__version__','unavailable'),'torch_version':torch.__version__}


def decode_asmc_subset(root, dq, positions):
    """Decode the focal pair in a retained 100-haplotype regional subset."""
    from asmc.asmc import ASMC, DecodingParams

    params = DecodingParams(
        in_file_root=str(root), dq_file=str(dq), out_file_root=str(root)+'_output',
        jobs=1, job_ind=1, decoding_mode_string='sequence',
        using_CSFS=True, compress=False, skip_CSFS_distance=0.0,
        no_batches=False, do_per_pair_posterior_mean=True,
        do_posterior_sums=False, do_major_minor_posterior_sums=False,
        do_per_pair_MAP=False,
    )
    # ASMC's CSFS projection uses the native C RNG, independently of NumPy.
    # Set the upstream fixed seed (1234) before Data/HMM construction.
    params.useKnownSeed = True
    params.batchSize = 64
    model = ASMC(params)
    model.set_store_per_pair_posterior_mean(True)
    model.set_store_per_pair_map(False)
    model.set_store_per_pair_posterior(False)
    model.set_store_sum_of_posterior(False)
    model.set_write_per_pair_posterior_mean(False)
    model.set_write_per_pair_map(False)

    expected = dict(jobs=1, jobInd=1, decodingSequence=True, usingCSFS=True,
                    compress=False, skipCSFSdistance=0.0, noBatches=False,
                    doPerPairPosteriorMean=True, doPosteriorSums=False,
                    doMajorMinorPosteriorSums=False, doPerPairMAP=False,
                    useKnownSeed=True, batchSize=64)
    actual = model.get_decoding_params()
    recorded = {name: getattr(actual, name) for name in expected}
    if recorded != expected:
        raise ValueError(f'ASMC native parameters differ from prescribed settings: {recorded}')
    if model.get_haploid_sample_size() != 100:
        raise ValueError('ASMC must load the retained 100 observed haplotypes')
    if not np.array_equal(model.get_physical_positions(), positions):
        raise ValueError('ASMC changed the regional position grid')
    expected_times = np.asarray(model.get_expected_times(), dtype=np.float64)
    if (expected_times.ndim != 1 or len(expected_times) < 2
            or not np.isfinite(expected_times).all() or np.any(expected_times <= 0)
            or np.any(np.diff(expected_times) <= 0)):
        raise ValueError('Invalid ASMC expected-time state grid')

    model.decode_pairs([0], [1])
    result = model.get_ref_of_results()
    if [(int(pair[0]), int(pair[2])) for pair in result.per_pair_indices] != [(0, 1)]:
        raise ValueError('ASMC decoded an unexpected focal haplotype pair')
    means = np.asarray(result.per_pair_posterior_means)
    if means.shape != (1, len(positions)) or not np.isfinite(means).all() or np.any(means <= 0):
        raise ValueError('Incomplete/nonpositive/nonfinite ASMC result')
    # The reference is owned by the native model. Copy only the requested mean
    # vector before this model is destroyed, without copying unused summaries.
    return means[0].copy(), {'native_parameters': recorded,
                            'native_csfs_seed': 1234,
                            'state_count': len(expected_times),
                            'expected_times_generations': expected_times.tolist()}


def asmc(a,folder):
    from importlib.metadata import version

    dq=Path(a.decoding_quantities or Path(a.tools_root)/'asmc_data/CEU_csfs50/decoding.decodingQuantities.gz')
    if not dq.is_file():raise FileNotFoundError(f'Missing ASMC decoding quantities {dq}')
    prior=Path(a.regional_root or a.output_root)/'regional'/f'{a.gene}_{a.pop}';require_complete(prior)
    z=np.load(prior/'results.npz');pairs=z['pairs'].tolist();data,pm,haps,source=chromosome(a)
    if not np.array_equal(z['population_hap_indices'],haps):raise ValueError('Regional population membership changed')
    gs,ge,mid,lo,hi=gene(a);G,pos=window_slice(data['G'],data['positions'],haps,lo,hi)
    helper=legacy('run_asmc_region.py');rng=np.random.default_rng(123);values=[];subsets=[];native_metadata=None
    asmc_version=version('asmc-asmc')
    if len(haps)<100:raise ValueError('The prespecified regional ASMC subset requires100 observed haplotypes; this is independent of the CSFS grid sample count')
    if not pairs or not len(pos):raise ValueError('Regional ASMC requires nonempty pairs and positions')
    for first,second in pairs:
        if first==second or min(first,second)<0 or max(first,second)>=len(haps):raise ValueError('Invalid regional ASMC focal pair')
        others=[i for i in range(len(haps)) if i not in (first,second)];pick=rng.choice(others,size=98,replace=False).tolist();sub=[first,second]+sorted(pick);subsets.append(sub)
        with tempfile.TemporaryDirectory(dir=folder,prefix='asmc-') as td:
            root=str(Path(td)/'data');helper.write_asmc_input(root,a.chr,pos,G[sub,:],50)
            value,metadata=decode_asmc_subset(root,dq,pos)
            if native_metadata is not None and metadata!=native_metadata:raise ValueError('ASMC native settings or state grid changed between subsets')
            native_metadata=metadata
            values.append(value)
    np.savez_compressed(folder/'results.npz',positions=pos,mean=np.asarray(values),pairs=np.asarray(pairs),subsets=np.asarray(subsets))
    return {'cache':source,'decoding_quantities':fingerprint(dq),'regional_dependency':fingerprint(prior/'COMPLETE.json'),'subset_seed':123,'diploid_samples':50,'csfs_haploid_sample_count':50,'observed_subset_convention':'100 observed haplotypes retained from historical regional approximation; independent of the CSFS grid sample count','asmc_version':asmc_version,'time_units':'generations from calibrated decoding quantities','no_pair_failures_ignored':True,**native_metadata}


def main():
    a=configure(args_parser().parse_args());folder,ident,done=begin(a)
    if done:print(f'Complete matching task: {folder}');return
    functions={'manifest':manifest,'selscan':selscan,'normalize':normalize,'h12':h12,'h12-aggregate':h12_aggregate,'neighborhood-background':neighborhood_background,'neighborhood-aggregate':neighborhood_aggregate,'variant':variant,'neighborhood':neighborhood,'xpehh':xpehh,'regional':regional,'asmc':asmc}
    start=time.time()
    try:details=functions[a.action](a,folder);details['elapsed_seconds']=time.time()-start;finish(folder,ident,details)
    except BaseException as exc:
        (folder/'FAILED.json').write_text(json.dumps({'error':str(exc),'type':type(exc).__name__},indent=2)+'\n');raise
    print(f'Completed {a.action}: {folder}',flush=True)

if __name__=='__main__':main()
