"""Render the reviewer diagnostics from completed, verified analysis tables."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from common import POPS, sha
from covariates import COVS, FOCAL

METRICS=['ihs_body','ihs_flank100','xp_body','xp_flank100']
LABELS=['iHS body','iHS flanks','XP-EHH body','XP-EHH flanks']
COLORS=['#0072B2','#0072B2','#D55E00','#D55E00']
MARKERS=['o','s','^','D']
STYLES=['-','--','-','--']

def title(ax,letter,label):
    ax.set_title(label,loc='left',pad=20,fontsize=9)
    ax.annotate(letter,xy=(-.16,1),xycoords='axes fraction',xytext=(0,20),textcoords='offset points',
                fontsize=11,fontweight='bold',va='baseline')

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.55)

def heatmap(fig,box,df,columns,labels,letter,heading,limit):
    ax=fig.add_axes(box)
    a=np.array([[df.loc[(df['pop']==p)&(df['column']==c),'rho'].iloc[0] for c in columns] for p in POPS])
    cmap=LinearSegmentedColormap.from_list('readable_diverging',plt.get_cmap('RdBu_r')(np.linspace(.18,.82,256)))
    im=ax.imshow(a,aspect='auto',vmin=-limit,vmax=limit,cmap=cmap,interpolation='none')
    for i,j in zip(*np.where(a<0)):
        ax.text(j,i,'−',ha='center',va='center',fontsize=6.5,color='black')
    ax.set_yticks(range(26),POPS,fontsize=6.5)
    ax.set_xticks(range(8),labels*2,rotation=45,ha='right',fontsize=7)
    ax.tick_params(length=0,pad=3)
    ax.axvline(3.5,color='white',lw=2)
    for y in [6.5,11.5,16.5,21.5]:ax.axhline(y,color='white',lw=.75)
    for s in ax.spines.values():s.set_visible(False)
    ax.text(.25,1.014,'Marginal',transform=ax.transAxes,ha='center',fontsize=7)
    ax.text(.75,1.014,'Partial',transform=ax.transAxes,ha='center',fontsize=7)
    title(ax,letter,heading)
    cax=fig.add_axes([box[0]+box[2]+.012,box[1]+.03,.013,box[3]-.06])
    cb=fig.colorbar(im,cax=cax,ticks=[-limit,0,limit]);cb.ax.tick_params(length=2,labelsize=7)
    cb.ax.set_ylabel('Spearman correlation',fontsize=7,labelpad=3)
    return ax

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((a.data/'comparison_verification.json').read_text())
    def read(name):
        file=a.data/f'{name}.csv'
        assert sha(file)==receipt['outputs'][file.name],f'Unverified table: {name}'
        return pd.read_csv(file)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.55,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
    fig=plt.figure(figsize=(7.2,9.5),facecolor='white')
    # A: all populations; identical complete-covariate set for each pair of estimates.
    cov=read('covariate_correlations');cov['column']=cov.kind+'_'+cov.covariate
    heatmap(fig,[.095,.695,.325,.244],cov,[k+'_'+c for k in ['marginal','partial'] for c in COVS],
        ['Length','Sites','Recomb.','BGS'],'a','Genomic covariate dependence',.7)
    # B: within-population quintiles; population range is explicitly descriptive.
    q=read('covariate_quintiles')
    axes=[]
    for i,(c,label) in enumerate(zip(COVS,['Gene length','Retained-site count','Recombination rate','BGS (B-prime)'])):
        ax=fig.add_axes([.63+(i%2)*.19,.835-(i//2)*.123,.15,.090]);axes.append(ax)
        g=q.loc[q.covariate==c].groupby('quintile').median_rank
        med=g.median()*100;low=g.min()*100;high=g.max()*100
        ax.fill_between(med.index,low,high,color='#0072B2',alpha=.17,lw=0)
        ax.plot(med.index,med,'o-',color='#0072B2',ms=2.7,lw=.9)
        ax.set_title(label,fontsize=7.5,pad=5);ax.set_ylim(0,100);ax.set_yticks([0,50,100]);ax.set_xticks([1,3,5])
        if i%2:ax.set_yticklabels([])
        else:ax.set_ylabel('Median rank (%)',labelpad=2)
        if i//2:ax.set_xlabel('Covariate quintile',labelpad=2)
        clean(ax)
    fig.text(.576,.964,'b',fontsize=11,fontweight='bold')
    fig.text(.63,.964,'Ranks across covariate strata',fontsize=9)
    fig.text(.63,.665,'Line: median across populations; band: range',fontsize=6.8)
    # C: agreement with low TMRCA, on the same eligible genes within each metric.
    h=read('haplotype_correlations');h=h.loc[(h['summary']=='fraction')&h.scope.isin(['common','partial'])&h.metric.isin(METRICS)].copy()
    h['column']=h.scope+'_'+h.metric
    lim=max(.3,np.ceil(h.rho.abs().max()*10)/10)
    heatmap(fig,[.095,.355,.325,.244],h,[k+'_'+m for k in ['common','partial'] for m in METRICS],
        ['iHS body','iHS flank','XP body','XP flank'],'c','Agreement with low TMRCA',lim)
    # D: availability is a separate outcome, not an implicit score of zero.
    av=read('availability_by_rank');ax=fig.add_axes([.63,.485,.34,.114])
    for i,m in enumerate(METRICS):
        g=av.loc[av.metric==m].sort_values('bin');x=g.bin.to_numpy()
        ax.fill_between(x,g.ci_low*100,g.ci_high*100,color=COLORS[i],alpha=.09,lw=0)
        ax.plot(x,g.fraction*100,color=COLORS[i],marker=MARKERS[i],ls=STYLES[i],ms=2.3,lw=.8,label=LABELS[i])
    ax.set_ylim(0,103);ax.set_yticks([0,50,100]);ax.set_xlim(-.35,10.35)
    ax.set_xticks([0,1,5,10],['<1','1-10','40-50','90-100']);ax.set_xlabel('TMRCA rank bin (%)',labelpad=2)
    ax.set_ylabel('Genes with a finite score (%)');clean(ax);title(ax,'d','Haplotype-score availability')
    ax.legend(ncol=2,frameon=False,loc='lower right',fontsize=6.7,handlelength=1.9,columnspacing=.65)
    # E: explicit reasons for missing iHS scores, pooled counts shown as descriptive.
    miss=read('haplotype_missingness');ax=fig.add_axes([.63,.356,.34,.052])
    states=['finite score','no MAF-eligible core','eligible core; no finite score'];cols=['#56B4E9','#DDDDDD','#999999']
    agg=miss.groupby('state')[['n','low1']].sum().reindex(states).fillna(0)
    values=np.array([agg.low1,agg.n-agg.low1],float);den=values.sum(axis=1);left=np.zeros(2)
    for j,(state,color) in enumerate(zip(states,cols)):
        v=100*values[:,j]/den;ax.barh([1,0],v,left=left,height=.55,color=color,edgecolor='white',lw=.4)
        for row in range(2):
            if v[row]>10:ax.text(left[row]+v[row]/2,1-row,f'{v[row]:.0f}%',ha='center',va='center',fontsize=6.5,color='black')
        left+=v
    ax.set_xlim(0,100);ax.set_yticks([1,0],['Lowest 1%','Other genes']);ax.set_xticks([0,50,100]);ax.set_xlabel('Gene-population observations (%)',labelpad=2)
    clean(ax);title(ax,'e','Why gene-body iHS is missing')
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(cols,['Finite score','No MAF-eligible core','Eligible core, no finite score'])],
        bbox_to_anchor=(.615,.322),loc='upper left',ncol=1,frameon=False,fontsize=6.7,handlelength=1,handleheight=.6,labelspacing=.3)
    # F: matched low-TMRCA vs control top-tail support with chromosome uncertainty.
    support=read('matched_support');ax=fig.add_axes([.16,.073,.305,.155])
    for group,offset,marker,fc in [(True,-.14,'o','#0072B2'),(False,.14,'s','white')]:
        for i,m in enumerate(METRICS):
            row=support.loc[(support.ihs_available==group)&(support.metric==m)].iloc[0]
            if np.isfinite(row.difference):
                x=row.difference*100;lo=row.ci_low*100;hi=row.ci_high*100
                ax.errorbar(x,i+offset,xerr=np.array([[x-lo],[hi-x]]),fmt=marker,mfc=fc,mec='#0072B2',ecolor='#0072B2',ms=4,lw=.8,capsize=2)
            else:ax.text(.97,i+offset,'N/A',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.5)
    ax.axvline(0,color='#888888',ls=':',lw=.7);ax.set_yticks(range(4),LABELS);ax.set_ylim(3.5,-.6)
    ax.set_xlabel('Excess top-5% support (percentage points)');clean(ax);title(ax,'f','Low-TMRCA genes vs matched controls')
    counts=support.loc[support.metric=='xp_body'].set_index('ihs_available').n_computable
    ax.text(0,1.045,f'Body iHS: {counts[True]:,} computable / {counts[False]:,} missing',
        transform=ax.transAxes,fontsize=6.8)
    ax.legend(handles=[Line2D([],[],marker='o',color='none',mec='#0072B2',mfc='#0072B2',label='Body iHS computable'),
        Line2D([],[],marker='s',color='none',mec='#0072B2',mfc='white',label='Body iHS missing')],
        loc='upper left',bbox_to_anchor=(-.15,-.31),ncol=2,frameon=False,fontsize=6.8,handlelength=1,columnspacing=.65)
    # G: selected examples: original/matched TMRCA ranks and four haplotype ranks.
    focal=read('focal_matched_ranks').set_index(['gene','pop']);fr=read('focal_haplotype_ranks').set_index(['gene','pop','metric'])
    labels=[f'{g} {pop}' for g,pop in FOCAL]
    arr=[]
    for g,pop in FOCAL:
        r=focal.loc[(g,pop)];arr.append([r.original_rank,r.matched_rank]+[fr.loc[(g,pop,m),'rank'] for m in METRICS])
    arr=np.array(arr,float)*100;ax=fig.add_axes([.69,.073,.28,.155])
    cmap=LinearSegmentedColormap.from_list('light_blues',plt.get_cmap('Blues')(np.linspace(.02,.55,256)))
    cmap.set_bad('#E1E1E1')
    im=ax.imshow(-np.log10(np.clip(arr/100,.0001,1)),vmin=0,vmax=3,cmap=cmap,aspect='auto',interpolation='none')
    for i in range(len(FOCAL)):
        for j in range(6):
            val=arr[i,j];text='NA' if not np.isfinite(val) else f'{val:.1f}' if val>=.1 else '<0.1'
            ax.text(j,i,text,ha='center',va='center',fontsize=6.5,color='black')
    ax.set_yticks(range(len(labels)),labels,fontsize=6.5);ax.set_xticks(range(6),['TMRCA','Matched','iHS body','iHS flank','XP body','XP flank'],rotation=60,ha='right',fontsize=6.5)
    ax.tick_params(length=0,pad=3);ax.axvline(1.5,color='white',lw=2)
    ax.axhline(7.5,color='white',lw=1)
    for s in ax.spines.values():s.set_visible(False)
    title(ax,'g','Focal rank percentiles')
    fig.text(.69,.008,'Lower = more extreme; gray = unavailable',fontsize=6.7)
    for ext in ['pdf','svg','png']:
        fig.savefig(a.output.with_suffix('.'+ext),dpi=450,facecolor='white')
    a.output.with_suffix('.json').write_text(json.dumps({'source_sha256':sha(__file__),
        'comparison_receipt_sha256':sha(a.data/'comparison_verification.json'),
        'width_inches':7.2,'height_inches':9.5,'minimum_font_points':6.5,
        'outputs':{ext:sha(a.output.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}},indent=2)+'\n')
    print(a.output.with_suffix('.pdf'))

if __name__=='__main__':main()
