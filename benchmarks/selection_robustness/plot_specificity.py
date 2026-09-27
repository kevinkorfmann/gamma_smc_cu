"""Publication-size plots from completed specificity analyses, with replicate CIs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS={'neutral':'#0072B2','fixed':'#D55E00','lost':'#777777'}


def band(values,rng,n=4000):
    means=values[rng.integers(len(values),size=(n,len(values)))].mean(axis=1)
    return values.mean(axis=0),np.quantile(means,[.025,.975],axis=0)


def plot(source,out,label):
    z=np.load(source/'profiles.npz');s=json.loads((source/'summary.json').read_text())
    rng=np.random.default_rng(4381636)
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':300})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
    x=(z['edges'][:-1]+z['edges'][1:])/2/1e6
    xb=(z['model_start']+z['model_end'])/2/1e6
    if not np.array_equal(z['model_end'][:-1],z['model_start'][1:]):
        raise ValueError('This staircase layout requires contiguous native bins')
    bedges=np.r_[z['model_start'],z['model_end'][-1]]/1e6
    masks={g:z['labels']==g for g in ['neutral','fixed']}
    for ax,key,ylabel in [(axes[0,0],'pi',r'Neutral diversity / $4N_e\mu$'),
                          (axes[0,1],'tmrca',r'Mean pairwise TMRCA / $2N_e$')]:
        ax.axhline(1,color='.75',ls='--',lw=.8)
        for g,mask in masks.items():
            if mask.sum()<2:raise ValueError('Too few replicates for curve')
            mean,ci=band(z[key][mask],rng)
            name='Neutral' if g=='neutral' else 'Fixed positive sweep'
            ax.plot(x,mean,color=COLORS[g],lw=1.2,label=f'{name} (n={mask.sum()})')
            ax.fill_between(x,*ci,color=COLORS[g],alpha=.16,lw=0)
        ax.set(ylabel=ylabel,ylim=(0,None))
    ymax=max(1.6,axes[0,0].get_ylim()[1],axes[0,1].get_ylim()[1])
    for ax in axes[0]:ax.set_ylim(0,ymax)
    axes[0,0].legend(loc='upper left',frameon=False)
    ax=axes[1,0]
    ax.stairs(z['published_B'],bedges,
              color='black',lw=1.7,label='Published parameters')
    for g,mask in masks.items():
        ax.stairs(z['Bprime'][mask].mean(axis=0),bedges,color=COLORS[g],lw=1.0,
                ls='--' if g=='neutral' else ':',label=f"Refit: {'neutral' if g=='neutral' else 'fixed sweep'}")
    ax.set(ylabel="Predicted B′",ylim=(0,1))
    ax.text(.03,.06,"Fixed inputs: ΔB′ = 0 exactly",transform=ax.transAxes,fontsize=8)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,.12))
    ax=axes[1,1]
    a=z['Bprime'][masks['fixed']];b=z['Bprime'][masks['neutral']]
    boot=a[rng.integers(len(a),size=(4000,len(a)))].mean(axis=1)-b[rng.integers(len(b),size=(4000,len(b)))].mean(axis=1)
    mean=a.mean(axis=0)-b.mean(axis=0);ci=np.quantile(boot,[.025,.975],axis=0)
    ax.axhline(0,color='.5',lw=.8);ax.stairs(mean*1e4,bedges,color=COLORS['fixed'],lw=1.2)
    ax.fill_between(bedges,np.r_[ci[0],ci[0,-1]]*1e4,np.r_[ci[1],ci[1,-1]]*1e4,
                    step='post',color=COLORS['fixed'],alpha=.2,lw=0)
    ax.set(ylabel="Refitted B′ difference (×10⁻⁴)")
    ax.text(.03,.95,'Fixed sweep − neutral',transform=ax.transAxes,fontsize=8,va='top')
    for letter,ax in zip('ABCD',axes.ravel()):
        ax.set_xlim(40.65,41.67);ax.set_xlabel('Chromosome 6 position (Mb, GRCh38)')
        ax.axvline(41.160846,color='.6',lw=.7,ls=':')
        ax.axvspan(41.158505,41.163186,color='#E69F00',alpha=.15,lw=0)
        ax.text(-.15,1.04,letter,transform=ax.transAxes,weight='bold',fontsize=11)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/f'specificity_{label}.{ext}')
    plt.close(fig)
    caption=(f"Positive-only specificity experiment using the {label} recombination map in simulations "
        "and the deposited deCODE coefficients for B′ prediction/refitting. "
        f"SLiM 4.3; diploid Ne=10,000; Q=1; 50 sampled haplotypes; s=0.1, h=0.5; "
        f"a single new mutation introduced 1,000 generations before sampling. "
        f"All {s['outcomes'].get('fixed',0)+s['outcomes'].get('lost',0)+s['outcomes'].get('segregating',0)} "
        f"positive attempts retained ({s['outcomes'].get('fixed',0)} fixed, {s['outcomes'].get('lost',0)} lost, "
        f"{s['outcomes'].get('segregating',0)} segregating), plus {s['outcomes']['neutral']} neutral replicates. "
        "A,B: independent-replicate means in 20-kb windows, with pointwise 95% bootstrap intervals. "
        "The selected site is excluded from neutral diversity. C: the deposited initial YRI CADD6/deCODE "
        "100-kb altgrid B′ fit and mean regional-injection refits; the curves nearly coincide. "
        "The model's functional annotations stay fixed even though neither simulated arm contains BGS. "
        "D: difference between mean refitted maps, conditional on fixation, with independent-replicate "
        "pointwise 95% bootstrap intervals. Both arms replace diversity in 29 full regional bins; "
        "all other whole-genome observations remain fixed. Scale is 10⁻⁴ B′ units. "
        "The vertical line marks the introduced allele; shading marks TREM2. These intervals quantify "
        "regional simulation uncertainty, not uncertainty in the whole-genome empirical fit. "
        "This is a controlled sensitivity experiment, not an empirical TREM2 selection posterior.\n")
    (out/f'specificity_{label}_caption.txt').write_text(caption)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--label',required=True)
    a=p.parse_args();plot(a.source,a.out,a.label)


if __name__=='__main__':main()
