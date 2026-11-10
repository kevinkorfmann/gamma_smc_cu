"""Render the prespecified benchmark summaries; no model fitting or rescoring."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS=['Gamma','iHS body','XP-EHH body']
LABELS=['Gamma-SMC (CPU)','iHS','XP-EHH']
COLORS=['#173F68','#5098C5','#8295A6']
MARKERS=['o','s','^']
REGIMES=['incomplete','near_fixed','completed','older_completed']
REGIME_LABELS=['Incomplete','Near fixation','Completed','Older completed']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':9,'axes.titleweight':'normal','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':8,
        'ytick.labelsize':9,'legend.fontsize':9,'text.color':'black',
        'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})

def panel(ax,data,width,endpoint,metric,title,letter,labels):
    block=data[(data.width==width)&(data.endpoint==endpoint)]
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.09,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    for j,(method,color,marker) in enumerate(zip(METHODS,COLORS,MARKERS)):
        rows=block[block.method==method].set_index('regime').loc[REGIMES]
        y=np.arange(4)+(j-1)*.19
        if metric=='rate':
            x=rows.rate.to_numpy()*100;lo=rows.ci_low.to_numpy()*100;hi=rows.ci_high.to_numpy()*100
        elif metric=='coverage':
            x=rows.coverage.to_numpy()*100;lo=rows.coverage_low.to_numpy()*100;hi=rows.coverage_high.to_numpy()*100
        else:
            x=rows.localisation_median.to_numpy()/1000;lo=rows.localisation_q25.to_numpy()/1000;hi=rows.localisation_q75.to_numpy()/1000
        ax.errorbar(x,y,xerr=np.maximum(0,np.array([x-lo,hi-x])),fmt=marker,color=color,
            markeredgecolor='black',markeredgewidth=.4,markersize=4.5,
            elinewidth=1.1,capsize=2,zorder=3)
    ax.set_yticks(range(4),REGIME_LABELS if labels else ['']*4)
    ax.set_ylim(3.5,-.5);ax.tick_params(axis='y',length=0,pad=6)
    ax.grid(axis='x',color='#e6e6e6',linewidth=.5,zorder=0)
    if metric=='localisation':
        ax.set_xlim(-3,200);ax.set_xticks([0,50,100,150,200])
        ax.set_xlabel('Distance from true target (kb)')
    else:
        ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel('Detected replicates (%)' if metric=='rate' else 'Computable replicates (%)')

def render(data,out,supplementary=False):
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.5))
    fig.subplots_adjust(left=.205,right=.985,bottom=.12,top=.86,wspace=.19,hspace=.65)
    handles=[Line2D([],[],color=c,marker=m,linestyle='None',markeredgecolor='black',markeredgewidth=.4,markersize=5,label=l) for c,m,l in zip(COLORS,MARKERS,LABELS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.58,.985),ncol=3,frameon=False,handletextpad=.4,columnspacing=1.4)
    if supplementary:
        configs=[(5000,'scan','rate','Scan detection: 5-kb windows'),(20000,'scan','rate','Scan detection: 20-kb windows'),(20000,'target','coverage','Score available in 20-kb target'),(20000,'scan','localisation','Localisation: 20-kb windows')]
    else:
        configs=[(5000,'target','rate','Detection in 5-kb target'),(20000,'target','rate','Detection in 20-kb target'),(5000,'target','coverage','Score available in 5-kb target'),(5000,'scan','localisation','Localisation: 5-kb windows')]
    for i,(ax,args) in enumerate(zip(axs.flat,configs)):
        panel(ax,data,*args,'abcd'[i],i%2==0)
    fig.text(.205,.028,'100 independent replicates per history. Bars: 95% CI (a–c); middle 50% of errors (d).',fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(out.with_suffix('.'+ext),facecolor='white')
    plt.close(fig)

def run(a):
    setup();a.out.mkdir(parents=True,exist_ok=True)
    source=a.summary/'performance.csv';data=pd.read_csv(source)
    for method in METHODS:
        for width in [5000,20000]:
            for endpoint in ['target','scan']:
                b=data[(data.method==method)&(data.width==width)&(data.endpoint==endpoint)]
                assert set(b.regime)==set(REGIMES+['neutral'])
                assert (b[b.regime!='neutral'].n==100).all() and (b[b.regime=='neutral'].n==200).all()
    render(data,a.out/'fig_detection_benchmark')
    render(data,a.out/'fig_si_detection_benchmark',True)
    manifest={'source_sha256':sha(source),'plot_source_sha256':sha(__file__),
       'main_panels':['5 kb target power','20 kb target power','5 kb target coverage','5 kb scan localisation'],
       'supplementary_panels':['5 kb scan power','20 kb scan power','20 kb target coverage','20 kb scan localisation'],
       'intervals':'Wilson 95% CI for rates; interquartile range for localisation',
       'localisation_population':'All computable selected replicates, not conditional on detection',
       'methods':METHODS,'outputs':{p.name:sha(p) for p in a.out.glob('*detection_benchmark.*') if p.suffix in ['.pdf','.png','.svg']}}
    (a.out/'fig_detection_benchmark.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
