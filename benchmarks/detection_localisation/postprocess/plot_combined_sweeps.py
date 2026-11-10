"""One SI figure: raw, objectively chosen sweep examples and all-replicate performance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REGIMES=['incomplete','near_fixed','completed','older_completed']
NAMES=['Incomplete','Near fixation','Completed','Older completed']
COLORS=['#2166AC','#C65C2E','#208878']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    raw=a.data/'raw_examples';rec=json.loads((raw/'verification.json').read_text())
    for name,h in rec['outputs'].items():assert sha(raw/name)==h
    perf=pd.read_csv(a.data/'summary/performance.csv')
    profiles=pd.read_csv(a.data/'profiles/profiles_per_replicate.csv.gz')
    # Exact raw summaries reproduce the independently extracted 20-kb windows.
    for r in rec['records']:
        row=profiles[(profiles.id==r['id'])&(profiles.offset_bp==0)].iloc[0]
        for metric in ['true_tmrca','inferred_tmrca']:
            assert np.isclose(r['centre_'+metric],row[metric],rtol=1e-12)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'font.weight':'normal',
        'axes.titleweight':'normal','axes.labelweight':'normal','axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','text.color':'black','axes.labelcolor':'black'})
    fig=plt.figure(figsize=(7.6,8.0))
    topgrid=fig.add_gridspec(2,2,left=.13,right=.98,bottom=.41,top=.90,wspace=.24,hspace=.46)
    data={r:np.load(raw/(r+'.npz')) for r in REGIMES}
    # Common axes span every value shown, without smoothing or clipping points.
    tvals=[];hvals=[]
    for z in data.values():
        trees=z['tree_profile']; tvals.extend(trees[(trees[:,1]>-250000)&(trees[:,0]<250000),2])
        tvals.extend(z['inferred_tmrca'][np.abs(z['g_offset'])<=250000])
        hvals.extend(z['ihs'][(np.abs(z['i_offset'])<=250000)&np.isfinite(z['ihs'])])
        hvals.extend(z['xp'][(np.abs(z['x_offset'])<=250000)&np.isfinite(z['xp'])])
    tlim=(min(tvals)*.85,max(tvals)*1.15);hlim=(min(0,min(hvals))-.3,max(hvals)+.5)
    for i,regime in enumerate(REGIMES):
        inner=topgrid[i//2,i%2].subgridspec(2,1,height_ratios=[1.25,1],hspace=.08)
        top=fig.add_subplot(inner[0]);bottom=fig.add_subplot(inner[1],sharex=top);z=data[regime]
        tree=z['tree_profile'];sel=(tree[:,1]>-250000)&(tree[:,0]<250000);tree=tree[sel]
        edges=np.r_[tree[:,0],tree[-1,1]]/1000
        top.stairs(tree[:,2],edges,color='#222222',lw=.85,baseline=None)
        sel=np.abs(z['g_offset'])<=250000
        top.scatter(z['g_offset'][sel]/1000,z['inferred_tmrca'][sel],s=1.1,color=COLORS[0],alpha=.85,linewidths=0,zorder=3)
        for xp,yp,color in [('i_offset','ihs',COLORS[1]),('x_offset','xp',COLORS[2])]:
            keep=(np.abs(z[xp])<=250000)&np.isfinite(z[yp])
            bottom.scatter(z[xp][keep]/1000,z[yp][keep],s=1.2,alpha=.65,color=color,linewidths=0)
        top.set_yscale('log');top.set_ylim(*tlim);top.set_yticks([1000,3000,10000],['1,000','3,000','10,000']);top.minorticks_off()
        bottom.set_ylim(*hlim);bottom.set_yticks([0,3,6])
        for ax in [top,bottom]:
            ax.axvline(0,lw=.8,ls=':',color='#777777',zorder=0)
            ax.set_xlim(-250,250);ax.set_xticks([-200,-100,0,100,200]);ax.tick_params(length=2.5,pad=2)
            ax.grid(axis='y',color='#e8e8e8',lw=.45)
        top.tick_params(axis='x',bottom=False,labelbottom=False)
        if i%2==0:
            top.set_ylabel('TMRCA\n(generations)',labelpad=4)
            bottom.set_ylabel('Core score',labelpad=4)
        top.text(-.015,1.16,'abcd'[i],transform=top.transAxes,fontsize=11)
        top.text(.055,1.16,NAMES[i],transform=top.transAxes,fontsize=10)
        if i>=2:bottom.set_xlabel('Distance from selected site (kb)',labelpad=3)
    profile_handles=[Line2D([],[],color='#222222',lw=.9,label='True TMRCA'),
        Line2D([],[],color=COLORS[0],ls='',marker='.',markersize=4,label='Inferred TMRCA'),
        Line2D([],[],color=COLORS[1],ls='',marker='.',markersize=4,label='|iHS|'),
        Line2D([],[],color=COLORS[2],ls='',marker='.',markersize=4,label='XP-EHH')]
    fig.legend(handles=profile_handles,loc='upper center',bbox_to_anchor=(.55,.995),ncol=4,
        frameon=False,fontsize=8.5,handletextpad=.4,columnspacing=1.2)
    fig.text(.13,.945,'a–d  Individual sweep simulations, without smoothing',fontsize=9)
    # Three performance panels with shared history rows.
    grid=fig.add_gridspec(1,3,left=.22,right=.98,bottom=.075,top=.255,wspace=.19)
    methods=['Gamma','iHS body','XP-EHH body'];markers=['o','s','^']
    axes=[fig.add_subplot(grid[0,j]) for j in range(3)]
    for j,ax in enumerate(axes):
        for k,(method,color,mark) in enumerate(zip(methods,COLORS,markers)):
            rows=perf[(perf.width==5000)&(perf.endpoint==('scan' if j==2 else 'target'))&(perf.method==method)].set_index('regime').loc[REGIMES]
            if j==0:x=rows.rate.to_numpy()*100;lo=rows.ci_low.to_numpy()*100;hi=rows.ci_high.to_numpy()*100
            elif j==1:x=rows.coverage.to_numpy()*100;lo=rows.coverage_low.to_numpy()*100;hi=rows.coverage_high.to_numpy()*100
            else:x=rows.localisation_median.to_numpy()/1000;lo=rows.localisation_q25.to_numpy()/1000;hi=rows.localisation_q75.to_numpy()/1000
            y=np.arange(4)+(k-1)*.19
            ax.errorbar(x,y,xerr=[np.maximum(0,x-lo),np.maximum(0,hi-x)],fmt=mark,color=color,
                markersize=3.5,elinewidth=.8,capsize=1.5,markeredgecolor='black',markeredgewidth=.3,zorder=3)
        if j==2:
            rows=perf[(perf.width==5000)&(perf.endpoint=='scan')&(perf.method=='XP-EHH maximum')].set_index('regime').loc[REGIMES]
            x=rows.localisation_median.to_numpy()/1000;lo=rows.localisation_q25.to_numpy()/1000;hi=rows.localisation_q75.to_numpy()/1000
            ax.errorbar(x,np.arange(4)+.37,xerr=[x-lo,hi-x],fmt='^',color=COLORS[2],mfc='white',
                markersize=3.7,elinewidth=.65,capsize=1.5,markeredgewidth=.7,zorder=3)
            ax.set_xlim(-3,200);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('Error (kb)',labelpad=3)
        else:
            ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Replicates (%)',labelpad=3)
        ax.set_yticks(range(4),NAMES if j==0 else ['']*4);ax.set_ylim(3.55,-.5);ax.tick_params(axis='y',length=0,pad=5)
        ax.grid(axis='x',color='#e8e8e8',lw=.45);ax.tick_params(axis='x',length=2.5,pad=2)
        ax.text(-.02,1.10,'efg'[j],transform=ax.transAxes,fontsize=11)
        ax.text(.08,1.10,['Target detection','Score available','Localisation'][j],transform=ax.transAxes,fontsize=9)
    fig.text(.13,.34,'e–g  Performance across all 100 replicates per history (5-kb windows)',fontsize=9)
    handles=[Line2D([],[],color=c,marker=m,ls='',markersize=4,label=l) for c,m,l in zip(COLORS,markers,['Gamma-SMC (CPU)','iHS','XP-EHH'])]
    handles.append(Line2D([],[],color=COLORS[2],marker='^',mfc='white',ls='',markersize=4,label='XP-EHH maximum (g)'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.317),ncol=4,frameon=False,
        fontsize=7.8,columnspacing=1.1,handletextpad=.4)
    fig.text(.22,.017,'Bars: 95% confidence intervals (e,f); middle 50% of errors (g).',fontsize=8)
    a.out.mkdir(exist_ok=True,parents=True)
    for ext in ['pdf','png','svg']:fig.savefig(a.out/('fig_si_sweep_benchmark.'+ext),dpi=400,facecolor='white')
    # Assert no label or caption uses bold text and no figure text is cropped.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontweight() in ['normal','regular',400]
            box=text.get_window_extent(renderer)
            assert box.x0>=-1 and box.x1<=fig.bbox.width+1 and box.y0>=-1 and box.y1<=fig.bbox.height+1,(text.get_text(),box)
    plt.close(fig)
    receipt=dict(status='passed',source_sha256=sha(__file__),raw_receipt_sha256=sha(raw/'verification.json'),
        performance_sha256=sha(a.data/'summary/performance.csv'),examples=rec['records'],
        profile_display='Exact tree geometric means across 4950 focal pairs; site-level inferred geometric means and unbinned core scores; no smoothing or added noise',
        metric_display='Primary 5 kb target detection/availability and scan localisation; extra maximum XP-EHH localisation',
        tmrca_limits=tlim,haplotype_limits=hlim,bold_text=False,
        outputs={p.name:sha(p) for p in a.out.glob('fig_si_sweep_benchmark.*') if p.suffix in ['.pdf','.png','.svg']})
    (a.out/'fig_si_sweep_benchmark.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
