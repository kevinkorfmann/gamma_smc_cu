"""Position-resolved truth, Gamma estimates and paired bias, with context tracks."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from common import digest,write_json

COLORS=['#222222','#E64B35','#3C5488']
LABELS=['True TMRCA','Gamma-SMC-CU: auto','Gamma-SMC-CU: fixed']


def load_prediction(path, records):
    m=json.loads(path.with_name('metrics.json').read_text())
    with np.load(path) as z:
        np.testing.assert_array_equal(z['edges'],np.arange(0,10000001,10000))
        np.testing.assert_array_equal(z['pairs'],[(i,j) for i in range(50) for j in range(i)])
        values=np.stack([z[k].mean(axis=1) for k in ['truth','auto','fixed']])
        for i,k in enumerate(['auto','fixed'],1):values[i,z[k+'_coverage']<.999]=np.nan
        assert np.all(np.isfinite(values[0])) and np.all(values[0]>0)
    assert m['pairs']==1225 and m['genotype_shape'][0]==50
    records.append(dict(path=str(path),predictions_sha256=digest(path),metrics_sha256=digest(path.with_name('metrics.json')),
        input_tree_sha256=m['tree_sha256'],inference_source_sha256=m['source_sha256']))
    return values


def summarize(obs,ids,nfamilies,reference,paired,width,rng,B):
    factor=width//10000;nb=1000//factor
    x=obs.reshape(len(obs),3,nb,factor).mean(axis=-1)
    ref=reference[:,0].reshape(len(reference),nb,factor).mean(axis=-1)
    valid=np.all(np.isfinite(x),axis=(0,1))
    sums=np.zeros((nfamilies,3,nb));counts=np.bincount(ids,minlength=nfamilies)
    for k,i in enumerate(ids):sums[i]+=x[k]
    sums[:,:,~valid]=0
    w=rng.multinomial(nfamilies,np.ones(nfamilies)/nfamilies,size=B)
    denominators=w@counts;w=w[denominators>0];denominators=denominators[denominators>0]
    draws=(w@sums.reshape(nfamilies,-1)).reshape(len(w),3,nb)/denominators[:,None,None]
    point=sums.sum(axis=0)/len(obs)
    # Ratios compare inference and truth on the same samples and same spatial bins.
    bias=np.full((2,nb),np.nan);bias_draws=np.full((len(w),2,nb),np.nan)
    bias[:,valid]=np.log2(point[1:,valid]/point[0,valid])
    bias_draws[:,:,valid]=np.log2(draws[:,1:,valid]/draws[:,0:1,valid])
    lo=np.full_like(point,np.nan);hi=np.full_like(point,np.nan)
    lo[:,valid],hi[:,valid]=np.quantile(draws[:,:,valid],[.025,.975],axis=0)
    blo=np.full_like(bias,np.nan);bhi=np.full_like(bias,np.nan)
    blo[:,valid],bhi[:,valid]=np.quantile(bias_draws[:,:,valid],[.025,.975],axis=0)
    point[:,~valid]=np.nan
    if paired:
        assert len(ref)==nfamilies
        ref_point=counts@ref/counts.sum()
        ref_draws=(w@(counts[:,None]*ref))/denominators[:,None]
    else:
        wr=rng.multinomial(len(ref),np.ones(len(ref))/len(ref),size=len(w))
        ref_point=ref.mean(axis=0);ref_draws=wr@ref/len(ref)
    rlo,rhi=np.quantile(ref_draws,[.025,.975],axis=0)
    return dict(mean=point,lo=lo,hi=hi,bias=bias,bias_lo=blo,bias_hi=bhi,
        reference=ref_point,reference_lo=rlo,reference_hi=rhi,
        edges=np.arange(0,10000001,width),valid=valid)


def context(maps,intervals,edges):
    cumulative=np.r_[0,np.cumsum(maps['map_rate']*np.diff(maps['map_position']))]
    rate=np.diff(np.interp(edges,maps['map_position'],cumulative))/np.diff(edges)*1e8
    density=np.zeros(len(edges)-1)
    for lo,hi in intervals:
        density+=np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
    return rate,density/np.diff(edges)*100


def make_figure(group,stats,maps,intervals,left,output,pdf):
    fig=plt.figure(figsize=(6.5,6.05))
    grid=fig.add_gridspec(4,2,height_ratios=[2.4,1.25,.65,.4],left=.115,right=.98,bottom=.145,top=.815,wspace=.23,hspace=.29)
    all_hi=[]
    for width,span in [(50000,(0,10000000)),(10000,(4500000,5500000))]:
        d=stats[width];centers=(d['edges'][:-1]+d['edges'][1:])/2;use=(centers>=span[0])&(centers<span[1])
        all_hi.extend([np.nanmax(d['hi'][:,use]),np.max(d['reference_hi'][use])])
    ymax=np.ceil(max(all_hi)/5000)*5
    bias_bound=1.
    for d in stats.values():
        finite=np.r_[d['bias_lo'].ravel(),d['bias_hi'].ravel()]
        bias_bound=max(bias_bound,float(np.nanmax(np.abs(finite))))
    bias_bound=np.ceil(bias_bound*2)/2
    for col,(width,span,title) in enumerate([(50000,(0,10000000),'Whole simulated region · 50-kb bins'),(10000,(4500000,5500000),'TREM2 focal region · 10-kb bins')]):
        d=stats[width];centers=(d['edges'][:-1]+d['edges'][1:])/2;x=(centers+left)/1e6
        use=(centers>=span[0])&(centers<span[1]);limits=((left+span[0])/1e6,(left+span[1])/1e6)
        axes=[fig.add_subplot(grid[row,col]) for row in range(4)]
        for row,ax in enumerate(axes):
            ax.set_xlim(limits);ax.axvspan((left+4950000)/1e6,(left+5050000)/1e6,color='#E9ECF0',zorder=0,lw=0)
            ax.axvline((left+5000000)/1e6,color='#AEB6BF',ls=':',lw=.6)
            ax.tick_params(labelbottom=row==3)
            ax.set_title(chr(97+row*2+col),loc='left',fontsize=8.1,pad=4)
            ax.set_xticks([38,40,42,44,46] if col==0 else [40.8,41.,41.2,41.4,41.6])
        axes[0].text(.5,1.075,title,transform=axes[0].transAxes,ha='center',fontsize=7.1)
        axes[0].fill_between(x,d['reference_lo']/1000,d['reference_hi']/1000,color='#AEB6BF',alpha=.12,lw=0)
        axes[0].plot(x,d['reference']/1000,color='#929DA6',ls=':',lw=.9)
        for i in [2,1,0]:
            axes[0].fill_between(x,d['lo'][i]/1000,d['hi'][i]/1000,color=COLORS[i],alpha=.10,lw=0)
            axes[0].plot(x,d['mean'][i]/1000,color=COLORS[i],lw=.85,ls='--' if i==0 else '-')
        axes[0].set_ylim(0,ymax)
        axes[1].axhspan(-bias_bound,0,color='#F1F5FA',zorder=0,lw=0)
        axes[1].axhline(0,color='#69737D',ls='--',lw=.65)
        for i in [1,0]:
            axes[1].fill_between(x,d['bias_lo'][i],d['bias_hi'][i],color=COLORS[i+1],alpha=.12,lw=0)
            axes[1].plot(x,d['bias'][i],color=COLORS[i+1],lw=.8)
        axes[1].set_ylim(-bias_bound,bias_bound)
        rate,density=context(maps,intervals,d['edges'])
        axes[2].stairs(rate,(d['edges']+left)/1e6,color='#69737D',lw=.7,baseline=None)
        axes[2].set_ylim(0,max(1,1.08*np.max(rate[use])))
        axes[3].stairs(density,(d['edges']+left)/1e6,color='#00A087',lw=.55,fill=True,alpha=.45)
        axes[3].set_ylim(0,max(10,np.ceil(np.max(density[use])/10)*10))
        if col==0:
            axes[0].set_ylabel('Mean TMRCA\n(1,000 generations)')
            axes[1].set_ylabel('Bias: log2\n(inferred / true)')
            axes[2].set_ylabel('cM/Mb',fontsize=7)
            axes[3].set_ylabel('BGS target\n(%)',fontsize=6.5)
        else:
            axes[0].tick_params(labelleft=False);axes[1].tick_params(labelleft=False)
        axes[3].set_xlabel('Chromosome 6 position (Mb, GRCh38)')
    fig.text(.115,.97,group['title'],fontsize=9,ha='left')
    fig.text(.115,.938,group['subtitle'],fontsize=6.6,ha='left')
    fig.legend([Line2D([],[],color=COLORS[i],ls='--' if i==0 else '-',lw=1) for i in range(3)]+
        [Line2D([],[],color='#929DA6',ls=':',lw=1)],LABELS+[group['reference_label']],
        loc='upper center',bbox_to_anchor=(.555,.924),ncol=2,frameon=False,columnspacing=1.4,handlelength=2.3)
    fig.text(.5,.072,'Bias below 0 = underestimation; −1 = half the true mean TMRCA, +1 = twice. Ribbons: 95% founder-bootstrap intervals.',
        ha='center',fontsize=6.15)
    fig.text(.5,.037,group['caution'],ha='center',fontsize=6.15,color='#B24532' if group.get('failed_burn') else '#404040')
    stem=output/group['name']
    for ext in ['pdf','png','svg']:fig.savefig(str(stem)+'.'+ext,dpi=400)
    pdf.savefig(fig);plt.close(fig)


def make_comparison(groups,all_stats,output,zoom):
    """Compact matched-map view; unfinished stronger-BGS sweeps are excluded."""
    names=['spatial_annotated_gamma','spatial_density30_s1e2','spatial_original_neutral_fixed_sweep']
    selected=[next(g for g in groups if g['name']==n) for n in names]
    width=10000 if zoom else 50000
    span=(4500000,5500000) if zoom else (0,10000000)
    fig=plt.figure(figsize=(6.5,4.45))
    grid=fig.add_gridspec(3,3,height_ratios=[2.3,1.25,.55],left=.092,right=.988,bottom=.23,top=.76,wspace=.16,hspace=.26)
    headings=['Annotated selection targets','Expanded selection targets (30%)','Sweep at fixation']
    subtitles=['BGS only; 6 calibration founders','BGS only; 6 calibration founders','31 sweeps; 20 contributing founders']
    bounds=[];bias_bounds=[]
    for g in selected:
        d=all_stats[g['name']][width];centers=(d['edges'][:-1]+d['edges'][1:])/2;use=(centers>=span[0])&(centers<span[1])
        bounds.extend([np.nanmax(d['hi'][:,use]),np.max(d['reference_hi'][use])])
        bias_bounds.append(np.nanmax(np.abs(np.r_[d['bias_lo'][:,use].ravel(),d['bias_hi'][:,use].ravel()])))
    ymax=np.ceil(max(bounds)/5000)*5;limit=np.ceil(max(bias_bounds)*2)/2
    for col,g in enumerate(selected):
        d=all_stats[g['name']][width];left=g['region']['left'];centers=(d['edges'][:-1]+d['edges'][1:])/2
        use=(centers>=span[0])&(centers<span[1]);x=(centers+left)/1e6
        axes=[fig.add_subplot(grid[row,col]) for row in range(3)]
        for row,ax in enumerate(axes):
            ax.set_xlim((left+span[0])/1e6,(left+span[1])/1e6)
            ax.axvspan((left+4950000)/1e6,(left+5050000)/1e6,color='#E9ECF0',zorder=0,lw=0)
            ax.axvline((left+5000000)/1e6,color='#AEB6BF',ls=':',lw=.6)
            ax.set_title(chr(97+row*3+col),loc='left',fontsize=8.1,pad=4)
            ax.tick_params(labelbottom=row==2,labelleft=col==0)
            ax.set_xticks([40.8,41.2,41.6] if zoom else [38,42,46])
        axes[0].text(.5,1.20,headings[col],transform=axes[0].transAxes,ha='center',fontsize=6.9)
        axes[0].text(.5,1.10,subtitles[col],transform=axes[0].transAxes,ha='center',fontsize=5.9)
        axes[0].fill_between(x,d['reference_lo']/1000,d['reference_hi']/1000,color='#AEB6BF',alpha=.13,lw=0)
        axes[0].plot(x,d['reference']/1000,color='#929DA6',ls=':',lw=.85)
        for i in [2,1,0]:
            axes[0].fill_between(x,d['lo'][i]/1000,d['hi'][i]/1000,color=COLORS[i],alpha=.10,lw=0)
            axes[0].plot(x,d['mean'][i]/1000,color=COLORS[i],lw=.8,ls='--' if i==0 else '-')
        axes[0].set_ylim(0,ymax)
        axes[1].axhspan(-limit,0,color='#F1F5FA',zorder=0,lw=0)
        axes[1].axhline(0,color='#69737D',ls='--',lw=.65)
        for i in [1,0]:
            axes[1].fill_between(x,d['bias_lo'][i],d['bias_hi'][i],color=COLORS[i+1],alpha=.12,lw=0)
            axes[1].plot(x,d['bias'][i],color=COLORS[i+1],lw=.75)
        axes[1].set_ylim(-limit,limit)
        rate,_=context(g['maps'],g['intervals'],d['edges'])
        axes[2].stairs(rate,(d['edges']+left)/1e6,color='#69737D',lw=.65,baseline=None)
        axes[2].set_ylim(0,max(1,1.08*np.max(rate[use])))
        if col==0:
            axes[0].set_ylabel('Mean TMRCA\n(1,000 generations)')
            axes[1].set_ylabel('Bias: log2\n(inferred / true)')
            axes[2].set_ylabel('cM/Mb',fontsize=7)
    fig.text(.092,.966,'Selection troughs and TMRCA inference bias',fontsize=9)
    fig.legend([Line2D([],[],color=COLORS[i],ls='--' if i==0 else '-',lw=1) for i in range(3)]+
        [Line2D([],[],color='#929DA6',ls=':',lw=1)],LABELS+['No-selection / no-sweep truth'],
        loc='upper center',bbox_to_anchor=(.55,.95),ncol=2,frameon=False,columnspacing=1.2,handlelength=2.2)
    fig.text(.54,.156,'Chromosome 6 position (Mb, GRCh38) · '+('10-kb bins' if zoom else '50-kb bins'),ha='center',fontsize=7.5)
    fig.text(.54,.102,'Below zero: underestimation (−1 = half truth). Above zero: overestimation (+1 = twice truth).',ha='center',fontsize=6)
    fig.text(.54,.063,'BGS panels: exploratory calibration. Sweep panel: original neutral-background bank, conditional on fixation.',ha='center',fontsize=6)
    fig.text(.54,.029,'30% targets were deliberately expanded beyond annotation. Ribbons: 95% founder-bootstrap intervals.',ha='center',fontsize=6)
    stem=output/('selection_dips_bias_zoom' if zoom else 'selection_dips_bias_genome')
    for ext in ['pdf','png','svg']:fig.savefig(str(stem)+'.'+ext,dpi=400)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    for k in ['root','spatial','out','fonts']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=4000);a=p.parse_args()
    for f in a.fonts.glob('*.ttf'):font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.2,
        'axes.labelsize':7.5,'axes.titlesize':8.1,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.6,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.6,'ytick.major.size':2.6,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    a.out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((a.spatial/'completion.json').read_text());assert complete['complete'] and len(complete['results'])==36
    sm=json.loads((a.spatial/'manifest.json').read_text())
    old=a.root/'runs/factorial_frequency_v1/production_10Mb'
    cal=a.root/'runs/stronger_bgs_v1/calibration_v1'
    cm=json.loads((cal/'manifest.json').read_text())
    summary=json.loads((a.root/'results/stronger_bgs_v1/calibration/summary.json').read_text())
    assert sm['calibration_summary_sha256']==digest(a.root/'results/stronger_bgs_v1/calibration/summary.json')
    source_records=[];controls={};families={}
    for bg in ['neutral','bgs']:
        families[bg]=sorted(old.glob(bg+'_family_*'));assert len(families[bg])==30
        controls[bg]=np.stack([load_prediction(f/'control/gamma/final/predictions.npz',source_records) for f in families[bg]])
    groups=[]
    titles={'annotated_gamma':'Exons + conserved sequence · Gamma DFE','annotated_s1e3':'Exons + conserved sequence · s = −0.001',
        'density15_s1e3':'Expanded selection targets (15% coverage) · s = −0.001','density30_s1e3':'Expanded selection targets (30% coverage) · s = −0.001',
        'density30_s1e4':'Expanded selection targets (30% coverage) · s = −0.0001','density30_s1e2':'Expanded selection targets (30% coverage) · s = −0.01'}
    for model in cm['models']:
        name=model['name'];row=next(r for r in summary['results'] if r['model']==name)
        obs=np.stack([load_prediction(a.spatial/name/f'family_{i:03d}/gamma/final/predictions.npz',source_records) for i in range(6)])
        reg=Path(model['region']);r=json.loads((reg/'region.json').read_text());maps=np.load(reg/'maps.npz')
        groups.append(dict(name='spatial_'+name,title=titles[name],subtitle='BGS only · 6 calibration founders · 50 haplotypes and all 1,225 pairs per founder',
            reference_label='Neutral truth reference',reference=controls['neutral'],paired=False,obs=obs,ids=np.arange(6),nf=6,
            region=r,maps=maps,intervals=maps['selected_intervals'],failed_burn=not row['burn_gate_passed'],
            caution='Exploratory: targets deliberately expanded beyond annotation; longer equilibration is pending.' if not row['burn_gate_passed'] else
            'Exploratory: targets deliberately expanded beyond annotation to test stronger BGS; confirmation is pending.' if model['model']['synthetic'] else
            'Exploratory calibration; effects assigned to conserved sites are an assumption. No published B-prime map is re-estimated.'))
    base_maps=groups[0]['maps'];base_region=groups[0]['region']
    for bg in ['neutral','bgs']:
        obs=[];ids=[]
        for i,f in enumerate(families[bg]):
            for run in sorted(f.glob('sweep_*')):
                if not run.is_dir():continue
                pred=run/'gamma/f100/predictions.npz'
                if pred.exists():obs.append(load_prediction(pred,source_records));ids.append(i)
        groups.append(dict(name='spatial_original_'+bg+'_fixed_sweep',
            title='Sweep at fixation · '+('neutral background' if bg=='neutral' else 'original exon-only BGS background'),
            subtitle=f'Completed original bank · {len(obs)} fixed sweeps from {len(set(ids))} founders · 50 haplotypes / 1,225 pairs',
            reference_label='Matched no-sweep truth',reference=controls[bg],paired=True,obs=np.stack(obs),ids=np.array(ids),nf=30,
            region=base_region,maps=base_maps,intervals=base_maps['exons'] if bg=='bgs' else np.empty((0,2)),
            caution='Conditional on natural fixation; controls weighted by successful introductions within each founder.'))
    rng=np.random.default_rng(260930);records=[];arrays={};all_stats={}
    with PdfPages(a.out/'spatial_dips_and_biases_all_models.pdf') as pdf:
        for g in groups:
            stats={width:summarize(g['obs'],g['ids'],g['nf'],g['reference'],g['paired'],width,rng,a.bootstrap) for width in [50000,10000]}
            all_stats[g['name']]=stats
            make_figure(g,stats,g['maps'],g['intervals'],g['region']['left'],a.out,pdf)
            for width,d in stats.items():
                for k,v in d.items():arrays[f"{g['name']}_{width}_{k}"]=v
            focal=np.arange(1000)*10000+5000;use=abs(focal-5000000)<50000
            # Same 100-kb aggregation for the compact numeric check; pair/position means first, then founder clusters.
            x=g['obs'][:,:,use].mean(axis=2);counts=np.bincount(g['ids'],minlength=g['nf'])
            sums=np.zeros((g['nf'],3))
            for i,v in zip(g['ids'],x):sums[i]+=v
            w=rng.multinomial(g['nf'],np.ones(g['nf'])/g['nf'],size=a.bootstrap);den=w@counts;valid=den>0
            draws=(w[valid]@sums)/den[valid,None];point=x.mean(axis=0)
            ratios=draws[:,1:]/draws[:,0,None]
            records.append(dict(name=g['name'],title=g['title'],snapshots=len(x),contributing_founders=len(set(g['ids'])),
                resampled_founders=g['nf'],focal_true_mean=float(point[0]),focal_auto_mean=float(point[1]),focal_fixed_mean=float(point[2]),
                auto_over_truth=float(point[1]/point[0]),fixed_over_truth=float(point[2]/point[0]),
                auto_over_truth_ci95=np.quantile(ratios[:,0],[.025,.975]).tolist(),fixed_over_truth_ci95=np.quantile(ratios[:,1],[.025,.975]).tolist(),
                caution=g['caution'],paired_reference=g['paired']))
            print(json.dumps(records[-1]),flush=True)
    for zoom in [True,False]:make_comparison(groups,all_stats,a.out,zoom)
    assert all(r['inference_source_sha256']==sm['inference_sha256'] for r in source_records)
    np.savez_compressed(a.out/'profiles_and_bias.npz',**arrays)
    write_json(a.out/'summary.json',dict(source_sha256=digest(__file__),bootstrap=a.bootstrap,seed=260930,
        bias='log2 of ratio of ensemble mean inferred TMRCA to ensemble mean true TMRCA; paired founder bootstrap',
        bins_whole_region_bp=50000,bins_zoom_bp=10000,shaded_focal_interval_bp=[4950000,5050000],
        full_10Mb_Gamma_input_for_both_views=True,inference_uses_scalar_mean_recombination=True,
        stronger_bgs_confirmation_complete=False,groups=records,source_records=source_records))


if __name__=='__main__':main()
