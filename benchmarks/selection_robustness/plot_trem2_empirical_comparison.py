"""Observed TREM2 beside audited simulations, at two common physical scales."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from common import digest, interval_means, snp_cells, write_json

LEFT=36160846; CENTER=41160846; B=4000
EDGES=np.arange(0,10000001,10000); MID=(EDGES[:-1]+EDGES[1:])/2
FLANK=(abs(MID-5000000)>=3000000)&(abs(MID-5000000)<4500000)
FOCAL=abs(MID-5000000)<50000
CORAL='#E64B35'; NAVY='#3C5488'; TEAL='#00A087'; BLUE='#4DBBD5'


def spatial_context(maps,edges):
    cumulative=np.r_[0,np.cumsum(maps['map_rate']*np.diff(maps['map_position']))]
    return np.diff(np.interp(edges,maps['map_position'],cumulative))/np.diff(edges)*1e8


def bootstrap(obs,ids,nf,width,rng):
    k=width//10000
    x=obs.reshape(len(obs),2,1000//k,k).mean(-1)
    valid=np.all(np.isfinite(x),axis=(0,1))
    counts=np.bincount(ids,minlength=nf); sums=np.zeros((nf,2,x.shape[-1]))
    for i,v in zip(ids,x):sums[i]+=v
    sums[:,:,~valid]=0
    w=rng.multinomial(nf,np.full(nf,1/nf),size=B);den=w@counts;w=w[den>0];den=den[den>0]
    draw=(w@sums.reshape(nf,-1)).reshape(len(w),2,-1)/den[:,None,None]
    mean=sums.sum(0)/len(obs)
    edges=np.arange(0,10000001,width);mid=(edges[:-1]+edges[1:])/2
    fl=(abs(mid-5000000)>=3000000)&(abs(mid-5000000)<4500000)&valid
    assert fl.sum()==3000000//width
    norm=mean/mean[:,fl].mean(-1)[:,None]
    nd=draw/draw[:,:,fl].mean(-1)[:,:,None]
    bias=np.zeros(x.shape[-1]);bd=np.zeros((len(w),x.shape[-1]))
    bias[valid]=np.log2(mean[1,valid]/mean[0,valid])
    bd[:,valid]=np.log2(draw[:,1,valid]/draw[:,0,valid])
    out=dict(mean=mean,lo=np.quantile(draw,.025,axis=0),hi=np.quantile(draw,.975,axis=0),
        normalized=norm,nlo=np.quantile(nd,.025,axis=0),nhi=np.quantile(nd,.975,axis=0),
        bias=bias,blo=np.quantile(bd,.025,axis=0),bhi=np.quantile(bd,.975,axis=0),edges=edges,valid=valid)
    for key,value in out.items():
        if key not in ['edges','valid']:value[...,~valid]=np.nan
    return out


def figure(groups,maps,bprime,out,normalized):
    fig=plt.figure(figsize=(7.2,6.8))
    gs=fig.add_gridspec(4,4,height_ratios=[1.3,1.5,.8,.85],left=.088,right=.987,
        top=.825,bottom=.16,wspace=.34,hspace=.56)
    headings=['Observed TREM2','BGS: annotated targets','BGS: expanded targets','Sweep at fixation']
    subs=['IBS and YRI; 20 pairs each','9.30% coverage; Gamma DFE','30.02% coverage; s = −0.01','s = 0.1; h = 0.5; no BGS']
    allaxes=[]
    for col,g in enumerate(groups):
        axes=[]
        for row in range(4):
            ax=fig.add_subplot(gs[row,col]);axes.append(ax)
            span=(0,10000000) if row==0 else (4500000,5500000)
            ax.set_xlim((LEFT+span[0])/1e6,(LEFT+span[1])/1e6)
            ax.axvspan((CENTER-50000)/1e6,(CENTER+50000)/1e6,color='#E9ECF0',zorder=0,lw=0)
            ax.axvline(CENTER/1e6,color='#AEB6BF',ls=':',lw=.65,zorder=0)
            ax.set_xticks([38,42,46] if row==0 else [40.8,41.2,41.6])
            ax.set_title(chr(97+row*4+col),loc='left',fontweight='bold',pad=4,fontsize=8)
            if row in [0,1]:
                width=50000 if row==0 else 10000;d=g['stats'][width]
                x=(d['edges'][:-1]+d['edges'][1:])/2/1e6+LEFT/1e6
                y=d['normalized'] if normalized else d['mean']/1000
                for j in [0,1]:
                    color=[CORAL,NAVY][j] if col==0 else ['#222222',CORAL][j]
                    if col:
                        lo=d['nlo'][j] if normalized else d['lo'][j]/1000
                        hi=d['nhi'][j] if normalized else d['hi'][j]/1000
                        ax.fill_between(x,lo,hi,color=color,alpha=.13,lw=0)
                    ax.plot(x,y[j],color=color,lw=.9,ls='--' if col and j==0 else '-')
                ax.set_yscale('log')
                ax.set_ylim((.012,8) if normalized else (.15,160))
                ticks=[.02,.1,.5,1,5] if normalized else [.2,1,5,20,100]
                ax.set_yticks(ticks,labels=[str(v) for v in ticks]);ax.minorticks_off()
                if normalized:ax.axhline(1,color='#9099A0',ls=':',lw=.65,zorder=0)
                if col:ax.tick_params(labelleft=False)
                if col==0:ax.set_ylabel('Mean TMRCA / flanks' if normalized else 'Mean TMRCA\n(1,000 generations)',fontsize=7)
                if row==0:
                    ax.text(.5,1.28,headings[col],ha='center',transform=ax.transAxes,fontsize=7.2)
                    ax.text(.5,1.13,subs[col],ha='center',transform=ax.transAxes,fontsize=5.8)
                else:
                    ax.text(.02,.97,'TREM2 ± 500 kb',ha='left',va='top',transform=ax.transAxes,fontsize=6.2)
                    ax.annotate('TREM2',xy=(CENTER/1e6,.045 if normalized else .55),
                        xytext=(41.30,.021 if normalized else .25),fontsize=5.8,
                        arrowprops=dict(arrowstyle='-',lw=.5,color='#56616A'),color='#56616A')
            elif row==2:
                if col==0:
                    for pop,color,ls in [('CEU',TEAL,'-'),('YRI',NAVY,'--')]:
                        q=bprime[pop]
                        for t in q.itertuples():
                            ax.plot([t.start/1e6,t.end/1e6],[t.Bprime,t.Bprime],color=color,ls=ls,lw=1)
                    ax.set_ylim(0,1.03);ax.set_yticks([0,.5,1]);ax.set_ylabel('Published B′',fontsize=7)
                    ax.text(.53,1.10,'CEU —    YRI --',ha='center',fontsize=5.8,transform=ax.transAxes,color='#46515B')
                else:
                    d=g['stats'][10000];x=(d['edges'][:-1]+d['edges'][1:])/2/1e6+LEFT/1e6
                    ax.axhline(0,color='#7F8990',ls='--',lw=.65)
                    ax.fill_between(x,d['blo'],d['bhi'],color=CORAL,alpha=.13,lw=0)
                    ax.plot(x,d['bias'],color=CORAL,lw=.8)
                    ax.set_ylim(-3,1.5);ax.set_yticks([-2,0,1])
                    if col==1:ax.set_ylabel('Bias: log2\n(inferred / true)',fontsize=6.7)
                    else:ax.tick_params(labelleft=False)
            else:
                rate=spatial_context(maps,EDGES)
                ax.stairs(rate,(EDGES+LEFT)/1e6,fill=True,color=BLUE,alpha=.4,lw=0)
                ax.stairs(rate,(EDGES+LEFT)/1e6,color='#2F788A',lw=.8,baseline=None)
                use=abs(MID-5000000)<500000
                ax.set_ylim(0,np.ceil(rate[use].max()/5)*5)
                if col==0:ax.set_ylabel('deCODE map\n(cM/Mb)',fontsize=7)
                else:ax.tick_params(labelleft=False)
            if row==2:ax.tick_params(labelbottom=False)
        allaxes.extend(axes)
    fig.text(.088,.975,'Observed TREM2 and matched-map selection simulations',fontsize=10)
    fig.legend([Line2D([],[],color=CORAL),Line2D([],[],color=NAVY),
                Line2D([],[],color='#222222',ls='--'),Line2D([],[],color=CORAL)],
        ['Observed IBS','Observed YRI','Simulation truth','Simulation Gamma-SMC-CU'],
        loc='upper center',bbox_to_anchor=(.54,.96),ncol=4,frameon=False,fontsize=6.5,columnspacing=1.2)
    fig.text(.54,.906,'Top: full 10 Mb (50-kb bins). Second row: the same profiles at TREM2 (10-kb bins).',ha='center',fontsize=6.5)
    fig.text(.54,.108,'Chromosome 6 position (Mb, GRCh38) · Grey band: focal 100-kb measurement window',ha='center',fontsize=7)
    note='Each curve is divided by its own mean 3–4.5 Mb from TREM2; normalization removes global scale differences.' if normalized else 'Physical-generation calibration is applied throughout; empirical and simulated demography and pair samples differ.'
    fig.text(.54,.077,note,ha='center',fontsize=6)
    fig.text(.54,.049,'BGS: 6 calibration replicates per model. Sweep: 31 fixations from 20/30 replicates; 95% cluster-bootstrap bands.',ha='center',fontsize=5.9)
    fig.text(.54,.023,'Expanded targets are a sensitivity test. No empirical truth or bias is available. These scenarios are not fitted to TREM2.',ha='center',fontsize=5.9)
    stem=out/('trem2_observed_vs_simulated_shape' if normalized else 'trem2_observed_vs_simulated_generations')
    for ext in ['pdf','png','svg']:fig.savefig(str(stem)+'.'+ext,dpi=350)
    plt.close(fig)


def realization_figure(groups,out):
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.5))
    fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.79,wspace=.25,hspace=.38)
    titles=['Observed IBS and YRI','Annotated BGS: 6 runs','Expanded BGS: 6 runs','Fixed sweeps: 31 runs']
    for col,g in enumerate(groups):
        for row,width in enumerate([50000,10000]):
            ax=axes[row,col];d=g['stats'][width]
            x=(d['edges'][1:]+d['edges'][:-1])/2/1e6+LEFT/1e6
            ax.axvspan((CENTER-50000)/1e6,(CENTER+50000)/1e6,color='#E9ECF0',zorder=0,lw=0)
            ax.axhline(1,color='#929DA6',ls=':',lw=.65)
            if col==0:
                for j,c in enumerate([CORAL,NAVY]):ax.plot(x,d['normalized'][j],color=c,lw=.8)
            else:
                curves=g['pair_matched'].reshape(-1,1000//(width//10000),width//10000).mean(-1)
                center=(d['edges'][1:]+d['edges'][:-1])/2
                fl=(abs(center-5000000)>=3000000)&(abs(center-5000000)<4500000)
                curves=curves/curves[:,fl].mean(1)[:,None]
                for c in curves:ax.plot(x,c,color=CORAL,alpha=.22,lw=.55)
                ax.plot(x,curves.mean(0),color='#A62518',lw=.95)
            ax.set_xlim((LEFT/1e6,(LEFT+10000000)/1e6) if row==0 else ((CENTER-500000)/1e6,(CENTER+500000)/1e6))
            ax.set_xticks([38,42,46] if row==0 else [40.8,41.2,41.6])
            ax.set_yscale('log');ax.set_ylim(.005,12);ax.set_yticks([.01,.1,1,10],labels=['0.01','0.1','1','10']);ax.minorticks_off()
            ax.set_title(chr(97+4*row+col),loc='left',fontsize=8,fontweight='bold')
            if row==0:ax.text(.5,1.15,titles[col],transform=ax.transAxes,ha='center',fontsize=7)
            if col==0:ax.set_ylabel('Mean TMRCA / flanks',fontsize=7)
            else:ax.tick_params(labelleft=False)
    fig.text(.09,.96,'What individual simulations look like with the same number of pairs',fontsize=9.5)
    fig.text(.53,.895,'Gamma-SMC-CU: 20 pairs per observed population or simulated realization',fontsize=7,ha='center')
    fig.text(.53,.14,'Chromosome 6 position (Mb, GRCh38) · Full 10 Mb above; focal 1 Mb below',ha='center',fontsize=7)
    fig.text(.53,.09,'Thin coral: individual simulated runs. Dark red: their mean after each run is normalized to its own far flanks.',ha='center',fontsize=6)
    fig.text(.53,.052,'Observed IBS: coral; YRI: navy. Sweep runs share 20/30 evolved populations and are not independent replicates.',ha='center',fontsize=6)
    fig.text(.53,.018,'Pair count is matched; cohort size, demography and inference context still differ. No best-matching realization was selected.',ha='center',fontsize=6)
    for ext in ['pdf','png','svg']:fig.savefig(out/f'trem2_matched_pair_realizations.{ext}',dpi=350)
    plt.close(fig)


def main():
    root=Path.cwd();refs=root/'references/trem2_empirical_comparison_v1'
    out=root/'results/stronger_bgs_v1/empirical_comparison_v1';out.mkdir(exist_ok=True)
    empirical=root/'results/stronger_bgs_v1/empirical_context_10Mb_v1'
    completion=json.loads((empirical/'completion.json').read_text());assert completion['complete']
    prior=root/'results/stronger_bgs_v1/spatial_v4'
    provenance=json.loads((prior/'summary.json').read_text());assert provenance['time_units']=='physical generations'
    raw=np.load(prior/'profiles_and_bias.npz')
    cm=json.loads((root/'runs/stronger_bgs_v1/calibration_v1/manifest.json').read_text())
    region=Path(next(m for m in cm['models'] if m['name']=='annotated_gamma')['region'])
    maps=np.load(region/'maps.npz')
    for f in (root/'references/figure_fonts').glob('*.ttf'):font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    observed=[];records=[]; arrays={}
    for pop in ['IBS','YRI']:
        z=np.load(empirical/f'{pop}.npz');v,cov=interval_means(*snp_cells(z['positions0']),z['mean'],EDGES+LEFT)
        v[cov<.999]=np.nan;observed.append(v.mean(1));arrays[pop+'_pair_bin_means']=v
        records.append(dict(population=pop,sha256=digest(empirical/f'{pop}.npz'),pairs=20,
            haplotypes=len(z['population_hap_indices']),retained_sites=len(z['positions0'])))
    obs=np.array(observed);stats={}
    for width in [10000,50000]:
        k=width//10000;y=obs.reshape(2,1000//k,k).mean(-1);edges=np.arange(0,10000001,width);mid=(edges[1:]+edges[:-1])/2
        fl=(abs(mid-5000000)>=3000000)&(abs(mid-5000000)<4500000)
        assert np.isfinite(y[:,fl]).all()
        stats[width]=dict(mean=y,normalized=y/y[:,fl].mean(-1)[:,None],edges=edges,valid=np.isfinite(y).all(0))
    groups=[dict(name='observed',stats=stats)];sources=[]
    rng=np.random.default_rng(270927)
    for name in ['annotated_gamma','density30_s1e2','original_neutral_fixed_sweep']:
        prefix='spatial_'+name;chosen=[];ids=[];pair_matched=[]
        pair_indices=np.sort(np.random.default_rng(42).choice(1225,20,replace=False))
        for rec in provenance['source_records']:
            path=Path(rec['path'])
            if name.startswith('original'):
                take='production_10Mb/neutral_family_' in str(path) and '/gamma/f100/' in str(path)
                family=int(next(p for p in path.parts if p.startswith('neutral_family_')).split('_')[-1]) if take else None
            else:
                take=f'calibration_spatial_gamma_v1/{name}/' in str(path)
                family=int(next(p for p in path.parts if p.startswith('family_')).split('_')[-1]) if take else None
            if not take:continue
            assert digest(path)==rec['predictions_sha256']
            with np.load(path) as z:
                v=np.stack([z['truth'].mean(1),z['auto'].mean(1)])
                v[1]*=rec['legacy_auto_to_physical_generations_factor'];v[1,z['auto_coverage']<.999]=np.nan
                chosen.append(v);ids.append(family)
                pm=z['auto'][:,pair_indices].mean(1)*rec['legacy_auto_to_physical_generations_factor']
                pm[z['auto_coverage']<.999]=np.nan;pair_matched.append(pm)
            sources.append(rec)
        assert len(chosen)==(31 if name.startswith('original') else 6)
        nf=30 if name.startswith('original') else 6
        s={w:bootstrap(np.stack(chosen),np.array(ids),nf,w,rng) for w in [10000,50000]}
        for w in s:
            np.testing.assert_allclose(s[w]['mean'],raw[f'{prefix}_{w}_mean'][:2],rtol=1e-12,equal_nan=True)
            np.testing.assert_allclose(s[w]['bias'],raw[f'{prefix}_{w}_bias'][0],rtol=1e-12,equal_nan=True)
        groups.append(dict(name=name,stats=s,pair_matched=np.stack(pair_matched)))
        arrays[name+'_matched_pair_indices']=pair_indices
        arrays[name+'_matched_pair_raw_means']=np.stack(pair_matched)
    bprime={}
    for pop in ['CEU','YRI']:
        q=pd.read_csv(refs/f'buffalo_kern2024_{pop}_CADD6_100kb.tsv.gz',sep='\t')
        bprime[pop]=q[(q.chrom=='chr6')&(q.start<CENTER+500000)&(q.end>CENTER-500000)]
        assert len(bprime[pop])>0
    summary=[];scale_metrics=[]
    cumulative=np.r_[0,np.cumsum(maps['map_rate']*np.diff(maps['map_position']))]*100
    for g in groups:
        d=g['stats'][10000]
        row=dict(name=g['name'],focal100kb_physical_mean=np.nanmean(d['mean'][:,FOCAL],axis=1).tolist(),
            far_flank_physical_mean=np.nanmean(d['mean'][:,FLANK],axis=1).tolist(),
            focal100kb_over_far_flanks=np.nanmean(d['normalized'][:,FOCAL],axis=1).tolist(),
            minimum10kb_within_focal=np.nanmin(d['normalized'][:,FOCAL],axis=1).tolist())
        summary.append(row)
        for j,label in enumerate(['IBS','YRI'] if g['name']=='observed' else ['truth','Gamma']):
            y=d['normalized'][j]
            anchor=np.flatnonzero(FOCAL)[np.nanargmin(y[FOCAL])]
            for threshold in [.25,.5,.75]:
                if y[anchor]>=threshold:continue
                lo=anchor;hi=anchor
                while lo>0 and np.isfinite(y[lo-1]) and y[lo-1]<threshold:lo-=1
                while hi+1<len(y) and np.isfinite(y[hi+1]) and y[hi+1]<threshold:hi+=1
                span=EDGES[[lo,hi+1]]
                genetic=np.interp(span,maps['map_position'],cumulative)
                scale_metrics.append(dict(group=g['name'],curve=label,threshold=threshold,
                    anchor_rule='minimum 10-kb bin inside prespecified focal 100 kb',
                    interval0=(span+LEFT).tolist(),width_bp=int(span[1]-span[0]),
                    width_cM=float(genetic[1]-genetic[0]),minimum=float(y[anchor]),
                    uncertainty='descriptive threshold width of mean curve; no confidence claim'))
        for w in g['stats']:
            for key,v in g['stats'][w].items():arrays[f"{g['name']}_{w}_{key}"]=v
    for normalized in [True,False]:figure(groups,maps,bprime,out,normalized)
    realization_figure(groups,out)
    np.savez_compressed(out/'profiles.npz',**arrays)
    write_json(out/'scale_metrics.json',scale_metrics)
    write_json(out/'summary.json',dict(source_sha256=digest(__file__),bootstrap=B,seed=270927,
        normalization='Each ensemble mean curve divided by its own mean in |distance| 3 to 4.5 Mb; denominator recomputed per bootstrap draw',
        coordinate_convention='0-based half-open; empirical VCF positions explicitly shifted by -1',
        empirical_completion_sha256=digest(empirical/'completion.json'),empirical=records,
        groups=summary,simulation_sources=sources,empirical_ABC=False,
        presentation='No empirical confidence interval: 20 pairs are not independent population replicates',
        sweep_strength=.1,dominance=.5,SLiM_Q=1))
    write_json(out/'audit.json',dict(passed=True,empirical_prior_overlap_checked=True,
        simulation_means_match_audited_spatial_v4=True,bias_matches_audited_spatial_v4=True,
        normalization_bootstrap_denominator_recomputed=True,common_coordinates_bins=True,
        no_reestimation_of_published_Bprime=True,empirical_pair_number=20))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
