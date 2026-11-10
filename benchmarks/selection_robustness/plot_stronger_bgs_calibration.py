"""Show every calibration model and the focal burn-in diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from common import digest,write_json


def main():
    p=argparse.ArgumentParser()
    for k in ['summary','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fonts',type=Path);a=p.parse_args()
    if a.fonts:
        for f in a.fonts.glob('*.ttf'):font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.2,'axes.labelsize':7.5,'axes.titlesize':8.1,'xtick.labelsize':6.5,
        'ytick.labelsize':6.5,'legend.fontsize':6.6,'axes.linewidth':.6,
        'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.6,
        'ytick.major.size':2.6,'axes.spines.top':False,'axes.spines.right':False,
        'mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42})
    s=json.loads(a.summary.read_text());rows=s['results'];assert len(rows)==6
    labels=['Exons + conserved; Gamma DFE','Exons + conserved; s = −0.001',
        '15% synthetic target; s = −0.001','30% synthetic target; s = −0.001',
        '30% synthetic target; s = −0.0001','30% synthetic target; s = −0.01']
    fig,ax=plt.subplots(1,2,figsize=(6.5,3.25),sharey=True)
    fig.subplots_adjust(left=.315,right=.985,bottom=.235,top=.81,wspace=.24)
    y=np.arange(6)[::-1]
    for i,row in enumerate(rows):
        for metric,offset,color,marker in [('focal100kb_tmrca',.09,'#00A087','o'),('genome_tmrca',-.09,'#3C5488','s')]:
            r=row['effects'][metric];lo,hi=r['ci95'];v=r['ratio']
            ax[0].errorbar(v,y[i]+offset,xerr=[[v-lo],[hi-v]],fmt=marker,
                color=color,ms=3,lw=.8,capsize=1.5)
        r=row['burn_in']['focal100kb_tmrca'];lo,hi=r['ci95'];v=r['ratio']
        color='#69737D' if row['burn_gate_passed'] else '#E64B35'
        ax[1].errorbar(v,y[i],xerr=[[v-lo],[hi-v]],fmt='o' if row['burn_gate_passed'] else 'x',
            color=color,ms=3.5,lw=.8,capsize=1.5)
    ax[0].axvline(1,color='#AEB6BF',ls=':',lw=.7)
    ax[0].set(xlim=(0,1.04),xlabel='TMRCA / neutral',yticks=y,yticklabels=labels)
    ax[0].set_xticks([0,.25,.5,.75,1]);ax[0].set_title('a  BGS-only reduction',loc='left',pad=8)
    ax[1].axvspan(.8,1.2,color='#E9ECF0',zorder=0,lw=0)
    ax[1].axvline(1,color='#AEB6BF',ls=':',lw=.7)
    ax[1].set(xlim=(.4,1.85),xlabel='Focal TMRCA: 10Ne / 5Ne')
    ax[1].set_xticks([.5,1,1.5]);ax[1].tick_params(axis='y',left=False)
    ax[1].set_title('b  Burn-in stability',loc='left',pad=8)
    ax[0].set_ylim(-.5,5.5)
    for axy in ax:axy.axhline(3.5,color='#D9DDE1',lw=.5,zorder=0)
    handles=[Line2D([],[],marker='o',color='#00A087',lw=.8,ms=3),
        Line2D([],[],marker='s',color='#3C5488',lw=.8,ms=3),
        Line2D([],[],marker='x',color='#E64B35',lw=.8,ms=3.5)]
    fig.legend(handles,['Focal 100 kb','Full 10 Mb','Extend burn-in'],frameon=False,
        loc='upper center',bbox_to_anchor=(.63,.98),ncol=3,columnspacing=1.1,handlelength=1.5)
    fig.text(.5,.10,'6 founders per model; 30 independent neutral references; 95% founder-bootstrap intervals.',
        ha='center',fontsize=6.4)
    fig.text(.5,.053,'Shading: focal burn-in tolerance on the mean ratio. Synthetic targets are stress tests, not human annotations.',
        ha='center',fontsize=6.2)
    a.out.mkdir(parents=True,exist_ok=True);stem=a.out/'stronger_bgs_calibration'
    for ext in ['pdf','svg','png']:fig.savefig(str(stem)+'.'+ext,dpi=400)
    plt.close(fig)
    write_json(a.out/'figure_provenance.json',dict(source_sha256=digest(__file__),summary_sha256=digest(a.summary),
        exploratory_calibration=True,confirmation_complete=False,style='Arial/NPG, 6.5 inch, matched to main-text case studies'))


if __name__=='__main__':main()
