"""Separate genealogical effects from depletion of directly selected variants."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for name in ['run','targets','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sensitivity',type=Path);p.add_argument('--fonts',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.fonts:
        for f in a.fonts.glob('*.ttf'):font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey='col',layout='constrained')
    for j,key in enumerate(['GRK2_GIH','TREM2_IBS']):
        t=np.load(a.targets/key/'target.npz');v=np.load(a.run/key/'validation_best.npz')
        meta=json.loads((a.targets/key/'manifest.json').read_text());x=(t['edges'][:-1]+t['edges'][1:])/2e6
        for ax in axes[:,j]:
            ax.plot(x,t['observed'],color='#4DBBD5' if j==0 else '#3C5488',lw=1.6,label='Observed')
            ax.axvspan(*np.array(meta['gene_interval0'])/1e6,color='#F0F2F7',zorder=0)
            ax.set(yscale='log',ylabel='Mean pairwise TMRCA (generations)',xlabel=f"Chromosome {meta['chromosome']} position (Mb, GRCh38)",xlim=t['edges'][[0,-1]]/1e6)
        ax=axes[0,j]
        lo,hi=np.quantile(v['inferred'],[.1,.9],axis=0)
        ax.fill_between(x,lo,hi,color='#E64B35',alpha=.12,lw=0)
        for name,label,col,ls in [('inferred','Gamma: actual polymorphisms','#E64B35','-'),
            ('reporter','Gamma: neutral markers','#00A087','-.'),('truth','True genealogical TMRCA','#69737D','--')]:
            ax.plot(x,v[name].mean(0),color=col,ls=ls,lw=1.3,label=label)
        ax.set_title(key.replace('_',' · ')+' · variant versus genealogy effects',loc='left',fontsize=10)
        ax.legend(fontsize=8,loc='best',frameon=False)
        ax=axes[1,j];ax.plot(x,v['inferred'].mean(0),color='#E64B35',label='Q = 10; burn-in = 10Ne; n = 24')
        lo,hi=np.quantile(v['inferred'],[.1,.9],axis=0);ax.fill_between(x,lo,hi,color='#E64B35',alpha=.12,lw=0)
        if a.sensitivity:
            for arm,col,ls in [('Q5','#00A087','--'),('burn20','#69737D','-.')]:
                z=np.load(a.sensitivity/key/arm/'ensemble.npz')['inferred']
                label=f'Q = 5; burn-in = 10Ne; n = {len(z)}' if arm=='Q5' else f'Q = 10; burn-in = 20Ne; n = {len(z)}'
                ax.plot(x,z.mean(0),color=col,ls=ls,lw=1.2,label=label)
        ax.set_title('Frozen-winner sensitivity checks',loc='left',fontsize=10);ax.legend(fontsize=8,loc='best',frameon=False)
    fig.suptitle('Negative selection · mechanism and numerical sensitivity\nRibbons: 10–90% of Q10 validation replicates',fontsize=11)
    for ext in ['png','pdf','svg']:fig.savefig(a.out/f'mechanism_checks.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__':main()
