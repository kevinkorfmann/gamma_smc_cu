"""Clear 2x2 view of verified covariate strata, matching and focal ranks."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from common import sha
from covariates import FOCAL

METRICS=['ihs_body','ihs_flank100','xp_body','xp_flank100']
BLUE='#0072B2'
ORANGE='#D55E00'
GREEN='#009E73'

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3,width=.6)

def panel_title(fig,x,y,letter,text,subtitle=None):
    fig.text(x,y,letter,fontsize=12,fontweight='bold',va='baseline')
    fig.text(x+.04,y,text,fontsize=10,va='baseline')
    if subtitle:fig.text(x+.04,y-.03,subtitle,fontsize=8.5,va='baseline')

def strata(ax,q,spec):
    for cov,label,color,marker,style in spec:
        g=q.loc[q.covariate==cov].groupby('quintile').median_rank
        med=g.median()*100;lo=g.min()*100;hi=g.max()*100
        ax.fill_between(med.index,lo,hi,color=color,alpha=.13,lw=0)
        ax.plot(med.index,med,color=color,marker=marker,ls=style,ms=4,lw=1.3,label=label)
    ax.set_xlim(.8,5.2);ax.set_ylim(0,100)
    ax.set_xticks(range(1,6),['Lowest\n20%','20-40%','40-60%','60-80%','Highest\n20%'])
    ax.set_yticks([0,25,50,75,100])
    ax.set_xlabel('Genes ordered by each variable',labelpad=6)
    ax.set_ylabel('Median TMRCA rank (%)',labelpad=5)
    ax.axhline(50,color='#B8B8B8',ls=':',lw=.7,zorder=0)
    ax.legend(loc='upper left',frameon=False,fontsize=8.5,handlelength=2.3,labelspacing=.65)
    clean(ax)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((a.data/'comparison_verification.json').read_text())
    def read(name):
        f=a.data/f'{name}.csv'
        assert sha(f)==receipt['outputs'][f.name],f'Changed source table: {name}'
        return pd.read_csv(f)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
    fig=plt.figure(figsize=(7.6,6.8),facecolor='white')
    # All four original covariates; bands are population ranges, not CIs.
    q=read('covariate_quintiles')
    panel_title(fig,.045,.955,'a','Gene length and retained sites')
    ax=fig.add_axes([.12,.665,.345,.25])
    strata(ax,q,[('length_bp','Gene length',BLUE,'o','-'),
                  ('n_sites','Retained-site count',ORANGE,'s','--')])
    panel_title(fig,.55,.955,'b','Recombination and BGS')
    ax=fig.add_axes([.64,.665,.33,.25])
    strata(ax,q,[('recombination','Recombination rate',GREEN,'o','-'),
                  ('bgs','BGS: retained diversity (B-prime)',BLUE,'s','--')])
    fig.text(.52,.525,'For each variable and population, genes are split into five groups (~20% each).\n'
             'Lines: median across 26 populations; bands: population range',
             ha='center',va='bottom',fontsize=8.5,linespacing=1.4)
    # Exact original estimates and chromosome-jackknife intervals.
    panel_title(fig,.045,.482,'c','Strong signals in low-TMRCA genes',
                'Strong signal = top 5% of scores for that test')
    fig.text(.085,.431,'0 = same rate; bars = 95% confidence intervals',fontsize=8)
    support=read('matched_support');ax=fig.add_axes([.14,.155,.325,.258])
    for available,offset,marker,color,face in [(True,-.14,'o',BLUE,BLUE),
                                            (False,.14,'s',ORANGE,'white')]:
        for i,m in enumerate(METRICS):
            r=support.loc[(support.ihs_available==available)&(support.metric==m)].iloc[0]
            if np.isfinite(r.difference):
                x=r.difference*100;lo=r.ci_low*100;hi=r.ci_high*100
                ax.errorbar(x,i+offset,xerr=np.array([[x-lo],[hi-x]]),fmt=marker,
                    mfc=face,mec=color,ecolor=color,ms=5,lw=1,capsize=2.5,mew=1.1)
                if available and m=='ihs_body':
                    ax.text(18.5,i+offset,f'{r.case_rate*100:.0f}% vs {r.control_rate*100:.0f}%',
                            ha='left',va='center',fontsize=8)
            else:ax.text(29,i+offset,'No score',ha='center',va='center',fontsize=8)
    ax.axvline(0,color='#777777',ls=':',lw=.8)
    ax.set_xlim(-10,42);ax.set_xticks([-10,0,10,20,30,40])
    ax.set_yticks(range(4),['iHS body','iHS flanks','XP-EHH body','XP-EHH flanks'])
    ax.set_ylim(3.48,-.52);clean(ax)
    ax.set_xlabel('Extra genes with a strong signal\nper 100 genes, compared with controls',labelpad=5)
    counts=support.loc[support.metric=='xp_body'].set_index('ihs_available').n_computable
    fig.legend(handles=[
        Line2D([],[],marker='o',color='none',mec=BLUE,mfc=BLUE,ms=5,
               label=f'With a body iHS score (n = {counts[True]:,})'),
        Line2D([],[],marker='s',color='none',mec=ORANGE,mfc='white',ms=5,
               label=f'Without a body iHS score (n = {counts[False]:,})')],
        bbox_to_anchor=(.10,.069),loc='upper left',frameon=False,fontsize=8.5,
        handlelength=1.2,labelspacing=.55,borderaxespad=0)
    # All twelve examples; distinguish missing scores from unavailable matches.
    panel_title(fig,.55,.482,'d','Focal gene comparisons',
                'Rank percentile (%); lower is more extreme')
    focal=read('focal_matched_ranks').set_index(['gene','pop'])
    hap=read('focal_haplotype_ranks').set_index(['gene','pop','metric'])
    values=[]
    for gene,pop in FOCAL:
        r=focal.loc[(gene,pop)]
        values.append([r.original_rank,r.matched_rank]+[hap.loc[(gene,pop,m),'rank'] for m in METRICS])
    ranks=np.asarray(values)*100;ax=fig.add_axes([.668,.14,.307,.25])
    cmap=LinearSegmentedColormap.from_list('readable_blues',plt.get_cmap('Blues')(np.linspace(.02,.55,256)))
    cmap.set_bad('#E4E4E4')
    ax.imshow(-np.log10(np.clip(ranks/100,.0001,1)),vmin=0,vmax=3,
              cmap=cmap,aspect='auto',interpolation='none')
    for i in range(len(FOCAL)):
        for j in range(6):
            v=ranks[i,j]
            label=('NM' if j==1 else 'NA') if not np.isfinite(v) else (f'{v:.2f}' if v<1 else f'{v:.1f}')
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='black')
    ax.set_yticks(range(len(FOCAL)),[f'{g} {pop}' for g,pop in FOCAL],fontsize=8)
    ax.set_xticks([]);ax.tick_params(length=0,pad=4)
    for s in ax.spines.values():s.set_visible(False)
    for j,label in enumerate(['Atlas','Matched','Body','Flanks','Body','Flanks']):
        ax.text((j+.5)/6,1.025,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=8)
    for x,label in [(1/6,'TMRCA'),(.5,'iHS'),(5/6,'XP-EHH')]:
        ax.text(x,1.125,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=8.5)
    for x in [1.5,3.5]:ax.axvline(x,color='white',lw=3)
    ax.axhline(7.5,color='white',lw=2)
    fig.text(.586,.095,'NA: no score   |   NM: match unavailable',fontsize=8)
    fig.text(.586,.068,'Flanks: 100 kb on each side, excluding the gene.',fontsize=8)
    for ext in ['pdf','svg','png']:
        fig.savefig(a.output.with_suffix('.'+ext),dpi=450,facecolor='white')
    a.output.with_suffix('.json').write_text(json.dumps({'source_sha256':sha(__file__),
        'comparison_receipt_sha256':sha(a.data/'comparison_verification.json'),
        'layout':'2x2','width_inches':7.6,'height_inches':6.8,'minimum_font_points':8,
        'preserved_analysis':True,'panels':{'a':'length and site-count strata','b':'recombination and BGS strata',
          'c':'previous panel f: matched support','d':'previous panel g: all 12 focal examples'},
        'outputs':{ext:sha(a.output.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}},indent=2)+'\n')
    print(a.output.with_suffix('.pdf'))

if __name__=='__main__':main()
