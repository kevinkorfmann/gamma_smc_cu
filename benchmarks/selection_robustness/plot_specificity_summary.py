"""Outcome-aware six-panel specificity figure; preserves native B-prime bins."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

from common import digest, interval_means, write_json


COLORS={'neutral':'#0072B2','lost':'#94999C','fixed':'#D55E00'}
MAP_COLORS={'deCODE':'#343A40','YRI':'#AA65A2'}
VIEW=(40.65,41.67)


def mean_band(values,seed,n=4000):
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(n,len(values)))].mean(axis=1)
    return values.mean(axis=0),np.quantile(means,[.025,.975],axis=0)


def difference_band(z,seed):
    a=z['Bprime'][z['labels']=='fixed'];b=z['Bprime'][z['labels']=='neutral']
    rng=np.random.default_rng(seed)
    draws=a[rng.integers(len(a),size=(4000,len(a)))].mean(axis=1)
    draws-=b[rng.integers(len(b),size=(4000,len(b)))].mean(axis=1)
    return a.mean(axis=0)-b.mean(axis=0),np.quantile(draws,[.025,.975],axis=0)


def spatial_style(ax,show_xlabel,view):
    ax.set_xticks([40,40.5,41,41.5,42,42.5] if view[1]-view[0]>2 else [40.8,41.0,41.2,41.4,41.6])
    ax.set_xlim(*view)
    ax.axvspan(41.1,41.2,color='#E69F00',alpha=.08,lw=0,zorder=0)
    ax.axvline(41.160846,color='#777777',ls=':',lw=.8,zorder=1)
    if show_xlabel:ax.set_xlabel('Chromosome 6 position (Mb, GRCh38)')
    else:ax.tick_params(labelbottom=False)


def distributions(ax,rows,key,seed):
    rng=np.random.default_rng(seed);groups=['neutral','lost','fixed']
    values=[np.array([float(r[key]) for r in rows if (r['regime']=='neutral' if g=='neutral'
                           else r['regime']=='positive' and r['outcome']==g)]) for g in groups]
    for i,(g,v) in enumerate(zip(groups,values)):
        violin=ax.violinplot(v,positions=[i],widths=.72,showextrema=False,bw_method='scott')
        for body in violin['bodies']:
            body.set_facecolor(COLORS[g]);body.set_edgecolor(COLORS[g]);body.set_alpha(.19);body.set_linewidth(.7)
        ax.scatter(i+rng.uniform(-.22,.22,len(v)),v,s=9,color=COLORS[g],alpha=.5,
                   edgecolors='none',zorder=2,rasterized=True)
        mean,ci=mean_band(v,seed+10+i,n=10000)
        ax.errorbar(i,mean,yerr=np.array([[mean-ci[0]],[ci[1]-mean]]),fmt='o',
                    mfc='white',mec='#222222',ms=4.0,ecolor='#222222',elinewidth=1.35,
                    capsize=4,capthick=1.35,zorder=4)
    ax.set_xticks(range(3),[f'{g.capitalize()}\nn = {len(v)}' for g,v in zip(groups,values)])
    ax.set_xlim(-.55,2.55);ax.tick_params(axis='x',length=0)
    low=min(v.min() for v in values);high=max(v.max() for v in values);span=high-low
    ax.set_ylim(max(0,low-.08*span),high+.30*span)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--sensitivity',type=Path,required=True);p.add_argument('--regions',type=Path,required=True)
    p.add_argument('--effects',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scale',type=Path,help='Optional post-hoc scale_summary.json')
    p.add_argument('--view',choices=['full','zoom'],default='full')
    a=p.parse_args();z=np.load(a.primary/'profiles.npz');s=np.load(a.sensitivity/'profiles.npz')
    summary=json.loads((a.primary/'summary.json').read_text())
    with (a.primary/'replicates.csv').open() as f:rows=list(csv.DictReader(f))
    effects=json.loads(a.effects.read_text());effect=next(r for r in effects['effects'] if r['map']=='deCODE' and r['group']=='fixed')
    scale=json.loads(a.scale.read_text()) if a.scale else None
    view=(z['edges'][0]/1e6,z['edges'][-1]/1e6) if a.view=='full' else VIEW
    stem='specificity_scale_summary' if scale else 'specificity_summary'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'axes.titlesize':9.5,'axes.titleweight':'medium','axes.titlelocation':'left',
        'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})
    fig=plt.figure(figsize=(7.35,7.35))
    grid=fig.add_gridspec(3,2,height_ratios=[.85,1.5,1.55],left=.105,right=.985,
                          bottom=.085,top=.963,hspace=.44,wspace=.31)
    axes=[fig.add_subplot(grid[i,j]) for i in range(3) for j in range(2)]
    for letter,ax,title in zip('ABCDEF',axes,['Recombination landscape','Published B′ context',
        'Genealogies (deCODE simulations)',
        'Influence of calibration span' if scale else 'Change in refitted B′',
        'Influence of averaging window' if scale else 'Focal neutral diversity','Focal B′ after refitting']):
        ax.set_title(title,pad=8)
        ax.text(-.17,1.07,letter,transform=ax.transAxes,fontsize=11,weight='bold',va='bottom')
        ax.tick_params(direction='out',width=.7,length=3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    # Regional context, with exact physical averages solely for display.
    ax=axes[0]
    for label in ['deCODE','YRI']:
        maps=np.load(a.regions/label/'maps.npz')
        reg=json.loads((a.regions/label/'region.json').read_text())
        edges=np.unique(np.r_[np.arange(0,reg['length'],20000),reg['length']])
        means,coverage=interval_means(maps['map_position'][:-1],maps['map_position'][1:],maps['map_rate'],edges)
        np.testing.assert_allclose(coverage,1)
        ax.stairs(means[:,0]*1e8,(edges+reg['left'])/1e6,color=MAP_COLORS[label],
                  lw=1.0,ls='-' if label=='deCODE' else '--',label=label,baseline=None)
    # Limits use the displayed interval, not remote peaks outside it.
    ax.set_ylabel('Recombination\n(cM/Mb)');ax.set_ylim(bottom=0)
    ax.legend(frameon=False,loc='upper left',ncol=2,handlelength=1.5,columnspacing=1.0)
    bedges=np.r_[z['model_start'],z['model_end'][-1]]/1e6
    if not np.array_equal(z['model_end'][:-1],z['model_start'][1:]):raise ValueError('Native bins must be contiguous')
    axes[1].stairs(z['published_B'],bedges,color=MAP_COLORS['deCODE'],lw=1.5,baseline=None)
    axes[1].set(ylabel='B′',ylim=(.45,1.0),yticks=[.5,.75,1.0])
    axes[1].text(.025,.88,'Fixed inputs: ΔB′ = 0',transform=axes[1].transAxes,
                 va='top',fontsize=8,color='#555555')
    # Independent simulation distributions, including unsuccessful attempts.
    ax=axes[2];x=(z['edges'][:-1]+z['edges'][1:])/2/1e6
    ax.axhline(1,color='#B7BDC2',ls='--',lw=.8)
    visible=(x>=view[0])&(x<=view[1]);upper=1.5
    for i,g in enumerate(['lost','neutral','fixed']):
        mean,ci=mean_band(z['tmrca'][z['labels']==g],4381640+i)
        ax.plot(x,mean,color=COLORS[g],lw=1.5 if g=='fixed' else 1.1,
                ls='--' if g=='lost' else '-',label=g.capitalize(),zorder=3 if g=='fixed' else 2)
        if g!='lost':
            ax.fill_between(x,*ci,color=COLORS[g],alpha=.15,lw=0,zorder=1)
            upper=max(upper,float(ci[1,visible].max()))
    ax.set(ylabel=r'Mean pairwise TMRCA / $2N_e$',ylim=(0,upper*1.13))
    handles,labels=ax.get_legend_handles_labels();lookup=dict(zip(labels,handles))
    ax.legend([lookup[k] for k in ['Neutral','Lost','Fixed']],['Neutral','Lost','Fixed'],
              frameon=False,loc='upper left',ncol=3,columnspacing=.75,handlelength=1.3)
    # Magnified response, at native 100-kb resolution.
    ax=axes[3]
    ax.axhline(0,color='#8B9297',lw=.8)
    if scale:
        spans=scale['calibration_spans'];xx=np.array([r['bp']/1e6 for r in spans])
        yy=np.array([r['delta_B'] for r in spans])*1e4
        ci=np.array([r['ci95'] for r in spans]).T*1e4
        ax.errorbar(xx,yy,yerr=np.array([yy-ci[0],ci[1]-yy]),color=MAP_COLORS['deCODE'],
                     fmt='o-',ms=3.5,lw=1.2,capsize=3,capthick=.9)
        ax.set(xlabel='Regional span replaced in the fit (Mb)',xlim=(0,3.05),
               ylabel='ΔB′, fixed − neutral (×10⁻⁴)',ylim=(min(ci[0])-.7,.8))
        ax.text(.03,.97,'Rest of genome held fixed',transform=ax.transAxes,va='top',fontsize=8,color='#555555')
        ax.text(.03,.06,'2.9 Mb = 0.13% of fitting weight',transform=ax.transAxes,fontsize=8,color='#555555')
    else:
        mean,ci=difference_band(z,4381643)
        ax.fill_between(bedges,np.r_[ci[0],ci[0,-1]]*1e4,np.r_[ci[1],ci[1,-1]]*1e4,
                        step='post',color=MAP_COLORS['deCODE'],alpha=.12,lw=0)
        ax.stairs(mean*1e4,bedges,color=MAP_COLORS['deCODE'],lw=1.35,label='deCODE',baseline=None)
        other,_=difference_band(s,4381644)
        np.testing.assert_array_equal(z['model_start'],s['model_start'])
        ax.stairs(other*1e4,bedges,color=MAP_COLORS['YRI'],ls='--',lw=1.1,label='YRI sensitivity',baseline=None)
        ax.set_ylabel('ΔB′, fixed − neutral (×10⁻⁴)')
        ax.legend(frameon=False,loc='upper left',ncol=2,columnspacing=.7,handlelength=1.4)
        lo,hi=ax.get_ylim();ax.set_ylim(lo,max(hi,1.25))
    for i in range(3 if scale else 4):spatial_style(axes[i],i>=2,view)
    # Every primary-bank simulation appears once in each lower distribution.
    if scale:
        windows=[r for r in scale['averaging_windows'] if r['name']=='TREM2_body' or r['name'].startswith('centered_') or r['name']=='full_region']
        xx=np.array([r['bp']/1000 for r in windows]);yy=np.array([r['pi_ratio'] for r in windows])
        ci=np.array([r['pi_ci95'] for r in windows]).T
        ax=axes[4];ax.axhline(1,color='#8B9297',lw=.8,ls='--')
        ax.axvline(100,color='#AAAAAA',lw=.8,ls=':')
        ax.errorbar(xx,yy,yerr=np.array([yy-ci[0],ci[1]-yy]),color=COLORS['fixed'],fmt='o-',
                     ms=3.5,lw=1.2,capsize=3,capthick=.9)
        ax.set(xscale='log',xlabel='Averaging-window width (kb)',
               ylabel='Diversity ratio: fixed / neutral',ylim=(0,1.14),xlim=(3.5,4500))
        ax.set_xticks([10,100,1000],['10','100','1,000'])
        ax.set_yticks([0,.25,.5,.75,1])
        ax.text(.03,.97,'100 kb: published B′ resolution',transform=ax.transAxes,va='top',fontsize=8,color='#555555')
    else:
        distributions(axes[4],rows,'focal_pi_ratio',4381645)
        axes[4].set_ylabel(r'Neutral-site diversity / $4N_e\mu$')
    distributions(axes[5],rows,'focal_B',4381646)
    axes[5].set_ylabel('B′ in focal 100-kb bin')
    axes[5].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    for ax,key in ([(axes[5],'focal_B')] if scale else [(axes[4],'focal_pi_ratio'),(axes[5],'focal_B')]):
        change=effect[key+'_relative_percent_change']
        text=f'Fixed versus neutral: {change:.1f}%' if key=='focal_pi_ratio' else f'Fixed versus neutral: {change:.3f}%'
        ax.text(.025,.97,text,transform=ax.transAxes,va='top',fontsize=8.5,
                 color=COLORS['fixed'],weight='medium')
    fig.text(.545,.017,'Local diversity perturbation of a genome-wide fit; no purifying selection simulated.' if scale else
              'Points: independent replicates. Bars: means and 95% bootstrap intervals.',
              ha='center',fontsize=7.5,color='#555555')
    a.out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(a.out/f'{stem}.{ext}')
    plt.close(fig)
    caption=("Positive-selection specificity of the B′ track used in the TREM2 figure. "
        "A: real regional deCODE and Pyrho-YRI recombination maps, averaged over 20 kb for display; "
        "simulations use the full-resolution maps. B: deposited initial Buffalo–Kern YRI CADD6/deCODE "
        "altgrid prediction in native 100-kb bins. Fixed inputs/parameters imply exactly zero change in B′. "
        "C: true mean pairwise TMRCA from deCODE-map SLiM simulations, normalized by 2Ne. "
        "All 200 beneficial mutations were retained: 15 fixed and 185 were lost; 40 independent neutral "
        "simulations serve as controls. Curves are replicate means in 20-kb windows; shading is a pointwise "
        "95% bootstrap interval for the neutral and fixed groups. D: mean global-refit B′ difference "
        "between fixed sweeps and neutral controls, with pointwise 95% bootstrap intervals for deCODE "
        "and the independent YRI-map sensitivity shown as a dashed curve. The scale is 10⁻⁴ B′ units. "
        "B′ predictors retain the deposited deCODE coefficients in both map sensitivities. "
        "A–D share genomic coordinates; pale shading marks the focal 100-kb bin and the dotted line "
        "marks the introduced allele at chr6:41,160,846 (zero-based). E,F: every primary-bank replicate "
        "appears in its outcome group for neutral-site diversity and globally refitted B′ in the focal bin. "
        "Light violins show sampling densities; points show independent simulations; white-centered "
        "markers and black bars show means and 95% bootstrap intervals. Callouts compare ratios of "
        "ensemble means, not per-replicate ratios. Neither simulated arm contains BGS. "
        "Ne=10,000 diploids, Q=1, 50 sampled haplotypes, mu=1.29e-8, s=0.1, h=0.5; the new beneficial "
        "mutation is introduced 1,000 generations before sampling. The selected site is excluded from "
        "neutral diversity. Regional expected pair counts replace 29 full bins in the published "
        "whole-genome likelihood; other data, annotations and recombination predictors remain fixed. "
        "Uncertainty is conditional on that fixed calibration background. These are sampling distributions, "
        "not ABC posteriors or empirical evidence assigning the actual TREM2 selection mechanism.\n")
    if scale:
        caption=("Local sweep influence on a genome-wide B′ fit, with spatial-scale diagnostics. "
            "A: deCODE and Pyrho-YRI input maps, shown as 20-kb means; simulations use full-resolution maps. "
            "B: published initial Buffalo–Kern YRI CADD6/deCODE altgrid B′ in the 29 native 100-kb fitting bins. "
            "B′ is context, not a simulated purifying-selection process; fixed inputs/parameters imply ΔB′=0. "
            "C: true normalized mean pairwise TMRCA, showing the entire 3.016-Mb simulation interval. "
            "There are 40 neutral controls, 185 lost beneficial mutations and 15 successful fixations. "
            "Curves are ensemble means in 20-kb windows, with pointwise 95% intervals for neutral and fixed groups. "
            "A–C share genomic coordinates; shading marks the focal 100-kb bin; the dotted line marks the "
            "introduced allele at chr6:41,160,846 (zero-based). D: focal B′ difference after replacing expected "
            "diversity counts over successively wider regional spans (1,3,5,11,21,29 bins) and refitting global "
            "parameters; every observation outside the replaced span stays fixed. Even the widest replacement "
            "contributes only 0.13% of genome-wide pair-count weight. E: dilution of the simulated diversity "
            "depression with averaging-window width. Windows span the 4.681-kb TREM2 body, centered intervals "
            "of 10,50,100,200,500,1,000 and 2,000 kb, and the full 3.016-Mb region. This diversity ratio is "
            "not a BGS-specific B′ estimate. D,E are post-hoc diagnostics reusing the same 15 fixed and 40 "
            "neutral independent replicates at every width; error bars are 95% replicate-bootstrap intervals. "
            "F: focal B′ sampling distributions for every primary-bank replicate. Violins show densities; "
            "points show replicates; white-centered markers/black bars show means and 95% intervals. "
            "The callout is a ratio of ensemble means. Simulations use Ne=10,000 diploids, Q=1, 50 sampled "
            "haplotypes, mu=1.29e-8, and a new beneficial mutation with s=0.1,h=0.5 introduced 1,000 generations "
            "before sampling. No purifying selection is simulated. The selected site is excluded from neutral "
            "diversity. All uncertainty is conditional on the fixed rest of the empirical genome; these "
            "results do not test widespread positive-selection contamination, longer-chromosome convergence, "
            "or the actual TREM2 mechanism.\n")
    (a.out/f'{stem}_caption.txt').write_text(caption)
    write_json(a.out/f'{stem}_provenance.json',dict(source_sha256=digest(__file__),view=a.view,
        primary_profiles_sha256=digest(a.primary/'profiles.npz'),sensitivity_profiles_sha256=digest(a.sensitivity/'profiles.npz'),
        effects_sha256=digest(a.effects),primary_counts=summary['outcomes'],
        plotting_only=True,primary_results_changed=False,post_hoc_scale_diagnostics=bool(scale),
        scale_sha256=digest(a.scale) if a.scale else None,
        outputs={f'{stem}.{e}':digest(a.out/f'{stem}.{e}') for e in ['pdf','svg','png']}))


if __name__=='__main__':main()
