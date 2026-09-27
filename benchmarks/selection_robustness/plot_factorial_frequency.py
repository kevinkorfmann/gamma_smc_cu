"""Matched-map figure using the current main-text case-study visual style."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from common import digest, write_json

COLORS={'neutral':'#AEB6BF','bgs':'#00A087','sweep':'#E64B35','bgs_sweep':'#3C5488'}
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7.2,'axes.labelsize':7.5,'axes.titlesize':8.1,'axes.titleweight':'normal',
    'font.weight':'normal','text.color':'black','axes.labelcolor':'black','xtick.color':'black',
    'ytick.color':'black','axes.edgecolor':'black','xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'legend.fontsize':6.6,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.size':2.6,'ytick.major.size':2.6,'axes.spines.top':False,
    'axes.spines.right':False,'mathtext.fontset':'dejavusans','pdf.fonttype':42,
    'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--region',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fonts',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.fonts:
        for path in a.fonts.glob('*.ttf'):font_manager.fontManager.addfont(path)
    plt.rcParams.update(STYLE)
    summary=json.loads((a.analysis/'summary.json').read_text())
    z=np.load(a.analysis/'profiles.npz');maps=np.load(a.region/'maps.npz')
    region=summary['region'];x=(z['edges'][:-1]+z['edges'][1:])/2+region['left']
    view=(x>=40660846)&(x<41660846);xm=x[view]/1e6
    stages=['f25','f50','f75','f100'];columns=['25%','50%','75%','100% (fixation)']
    groups=[('neutral','control','Neutral',COLORS['neutral'],':'),
        ('bgs','control','Neutral + BGS',COLORS['bgs'],'--'),
        ('neutral',None,'Neutral + sweep',COLORS['sweep'],'-'),
        ('bgs',None,'Neutral + BGS + sweep',COLORS['bgs_sweep'],'-')]
    if any(f'{bg}_{stage}_truth_mean' not in z for bg in ['neutral','bgs'] for stage in ['control',*stages]):
        raise ValueError('Four-arm plot needs successful sweeps and controls in both backgrounds')
    for mode in ['auto','fixed']:
        fig=plt.figure(figsize=(6.5,5.15))
        grid=fig.add_gridspec(3,4,height_ratios=[1,1,.28],left=.105,right=.988,bottom=.12,top=.84,
            wspace=.18,hspace=.43)
        values=[]
        for method in ['truth',mode]:
            for stage in stages:
                for bg,base,*_ in groups:
                    key=f'{bg}_{base or stage}_{method}'
                    values.extend([z[key+'_lo'][view],z[key+'_hi'][view]])
        lo=max(1,float(np.nanmin(values))*.8);hi=float(np.nanmax(values))*1.15
        for row,method in enumerate(['truth',mode]):
            for col,stage in enumerate(stages):
                ax=fig.add_subplot(grid[row,col])
                for bg,base,label,color,ls in groups:
                    key=f'{bg}_{base or stage}_{method}'
                    ax.fill_between(xm,z[key+'_lo'][view],z[key+'_hi'][view],color=color,alpha=.12,lw=0)
                    ax.plot(xm,z[key+'_mean'][view],color=color,ls=ls,lw=.9)
                ax.axvspan(41.110846,41.210846,color='#F0F2F7',zorder=0,lw=0)
                ax.axvline(41.160846,color='#AEB6BF',lw=.45,ls=':',zorder=0)
                ax.set(xlim=(40.660846,41.660846),yscale='log',ylim=(lo,hi))
                ax.set_xticks([40.8,41.2,41.6]);ax.tick_params(labelbottom=(row==1))
                ax.set_title(chr(97+row*4+col),loc='left',pad=6)
                if row==0:ax.text(.52,1.09,columns[col],ha='center',transform=ax.transAxes,fontsize=7.2)
                if col==0:ax.set_ylabel(('True TMRCA' if row==0 else 'Gamma-SMC-CU')+'\n(generations)')
                else:ax.tick_params(labelleft=False)
        # A single common reference track rather than four redundant maps.
        for col in range(1):
            ax=fig.add_subplot(grid[2,:])
            ax.stairs(maps['map_rate']*1e8,(maps['map_position']+region['left'])/1e6,
                baseline=None,color='#69737D',lw=.65)
            ax.set(xlim=(40.660846,41.660846),ylim=(0,22))
            # Exact coding-interval rug shares the genomic axis.
            for low,high in maps['exons']:
                low=(low+region['left'])/1e6;high=(high+region['left'])/1e6
                if high>=40.660846 and low<=41.660846:
                    ax.plot([low,high],[-3,-3],color='#AEB6BF',lw=1.2,clip_on=True)
            ax.set_ylim(-4,22);ax.set_yticks([0,20]);ax.set_xticks([40.8,41.2,41.6])
            if col==0:ax.set_ylabel('cM/Mb',fontsize=6.6)
            else:ax.tick_params(labelleft=False)
            ax.set_title('Shared recombination map and exon intervals',loc='left',fontsize=6.6,pad=5)
            ax.axvspan(41.110846,41.210846,color='#F0F2F7',zorder=0,lw=0)
        fig.legend([Line2D([],[],color=c,ls=ls,lw=1.1) for _,_,_,c,ls in groups],
            [v[2] for v in groups],loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,
            frameon=False,columnspacing=1.5,handlelength=2.0)
        sizes={r['background']:r for r in summary['counts'] if r['stage']=='f100'}
        fig.text(.55,.905,
            f"Attained stages: {sizes['neutral']['snapshots']} neutral-background sweeps / {sizes['neutral']['contributing_families']} founders; "
            f"{sizes['bgs']['snapshots']} BGS-background sweeps / {sizes['bgs']['contributing_families']} founders",
            ha='center',fontsize=6.2)
        fig.text(.55,.045,'Chromosome 6 position (Mb, GRCh38)',ha='center',fontsize=7.5)
        if not summary['complete']:fig.text(.98,.006,'Preliminary: incomplete bank',ha='right',fontsize=6.5)
        stem=a.out/f'factorial_frequency_gamma_{mode}'
        for ext in ['pdf','svg','png']:fig.savefig(str(stem)+'.'+ext,dpi=400)
        plt.close(fig)
    # Compact contrasts make the effect size and recovery easier to read than
    # overlapping genomic profiles alone. Each sweep is normalized to its own
    # no-sweep background; the genomic panels show absolute BGS differences.
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.95))
    fig.subplots_adjust(left=.105,right=.985,bottom=.21,top=.74,wspace=.30)
    lookup={(r['background'],r['stage'],r['method']):r for r in summary['effects']}
    for ax,ss,xx,letter,xlabel in [
        (axes[0],stages,[25,50,75,100],'a','Selected allele frequency (%)'),
        (axes[1],['f100','fixed_plus250','fixed_plus1000'],[0,250,1000],'b','Generations after fixation')]:
        for bg,color in [('neutral',COLORS['sweep']),('bgs',COLORS['bgs_sweep'])]:
            for method,ls,marker in [('truth','--','o'),('auto','-','s')]:
                rows=[lookup[(bg,s,method)] for s in ss]
                yy=np.array([r['focal100kb_retained_tmrca'] for r in rows])
                ci=np.array([r['ci95'] for r in rows])
                ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.10,lw=0)
                ax.plot(xx,yy,color=color,ls=ls,marker=marker,markersize=3,lw=.95)
        if letter=='a':ax.axhline(1,color='#69737D',ls=':',lw=.6)
        ax.set(xlabel=xlabel,ylabel='Focal TMRCA / no-sweep control')
        ax.set_xticks(xx);ax.set_ylim(bottom=0);ax.set_title(letter,loc='left',pad=6)
        ax.text(.5,1.025,'Sweep progression' if letter=='a' else 'Recovery (expanded scale)',
            ha='center',transform=ax.transAxes,fontsize=7.2)
        if letter=='b':
            ax.set_ylim(0,.13);ax.set_yticks([0,.04,.08,.12])
    fig.legend([Line2D([],[],color=COLORS['sweep']),Line2D([],[],color=COLORS['bgs_sweep']),
                Line2D([],[],color='black',ls='--',marker='o',markersize=3),
                Line2D([],[],color='black',ls='-',marker='s',markersize=3)],
        ['Sweep on neutral background','Sweep on BGS background','True TMRCA','Gamma-SMC-CU (auto)'],
        loc='upper center',ncol=2,frameon=False,fontsize=6.6)
    control=next(r for r in summary['background_control_contrasts'] if r['method']=='truth')
    fig.text(.55,.827,
        f"BGS-only / neutral focal TMRCA: {control['focal100kb_retained_tmrca']:.2f} "
        f"(95% CI {control['ci95'][0]:.2f}–{control['ci95'][1]:.2f}); "
        f"fixed sweeps: {summary['neutral']['outcomes']['fixed']} neutral, {summary['bgs']['outcomes']['fixed']} BGS",
        ha='center',fontsize=6.2)
    for ext in ['pdf','svg','png']:fig.savefig(a.out/f'factorial_frequency_effects.{ext}',dpi=400)
    plt.close(fig)
    caption=(
        'Matched-map BGS × sweep simulations at TREM2. Columns show first natural crossing of '
        '25%, 50%, 75%, and 100% allele frequency. Rows show true mean pairwise TMRCA and '
        'Gamma-SMC-CU posterior means, averaged over the same 1,225 pairs of 50 sampled haplotypes. '
        'Controls are continuations without a beneficial introduction. All arms use the same '
        '10-Mb deCODE map; the displayed region is 1 Mb. Lower tracks show the shared recombination '
        'map and coding intervals once; grey shading marks the centred 100-kb summary interval. '
        'BGS is generated by the Kim et al. gamma DFE in coding intervals. '
        'Ribbons are 95% pointwise founder-cluster bootstrap intervals, not posterior credible bands. '
        'The auto and fixed files use default data-estimated scaling and known-generative scaling, '
        'respectively. Gamma-SMC-CU receives the common scalar mean recombination rate. Stage means '
        'are conditional on attainment; all failed introductions remain in the reported outcome counts. '
        'The companion contrast figure averages over the focal centred 100 kb and divides by the '
        'corresponding no-sweep background; post-fixation ages have their own axis with an expanded y scale. '
        'Connecting lines guide the eye between observed stages and do not interpolate a fitted trajectory. '
        'Exact stage and founder sample sizes are in summary.json. This is not a re-estimated published B′ map.')
    (a.out/'caption.txt').write_text(caption+'\n')
    write_json(a.out/'provenance.json',dict(plot_sha256=digest(__file__),
        analysis_sha256=digest(a.analysis/'summary.json'),maps_sha256=digest(a.region/'maps.npz'),
        style_reference='current main-text Figures 4–5 gen_fig_case_studies.py and figure_palette.py',
        palette=COLORS,font=font_manager.findfont('Arial',fallback_to_default=True),
        width_inches=6.5,scientific_visual_qa_pending=True))


if __name__=='__main__':main()
