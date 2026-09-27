#!/usr/bin/env python3
"""Make a paginated locus atlas and a two-locus TMRCA/recombination comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text

DARK='#173F66';LIGHT='#6BAED6';PALE='#D8EAF5'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42})


def load(root,gene,pop,window=10000):
    d=pd.read_csv(root/'maps'/f'{gene}.{pop}.windows.tsv',sep='\t')
    return d[d.window_bp==window].copy()


def draw_rate(ax,root,t,split=False):
    for i,pop in enumerate(t.populations.split(',')):
        color=DARK if i==0 else LIGHT;style='-' if i==0 else '--'
        d=load(root,t.gene,pop);values=d.cM_Mb_Ne10000.where(d.coverage_fraction>=.99)
        x=(d.start+d.end)/2e6+1e-6
        ax.plot(x,values,color=color,ls=style,lw=1,label=pop)
        if split:
            for half in [1,2]:
                p=root/'maps'/f'{t.gene}.{pop}_half{half}.windows.tsv'
                if p.exists():
                    h=pd.read_csv(p,sep='\t').query('window_bp==10000')
                    ax.plot((h.start+h.end)/2e6+1e-6,h.cM_Mb_Ne10000.where(h.coverage_fraction>=.99),
                            color=color,alpha=.30,lw=.55,ls=':')
    ax.axhline(1,color='.45',lw=.7,ls=(0,(2,2)),label='Gamma-SMC input')
    ax.axvspan((t.gene_start0+1)/1e6,t.gene_end0/1e6,color=PALE,zorder=0)
    ax.set(xlim=((t.display_start0+1)/1e6,t.display_end0/1e6),yscale='log',
           ylabel='Rate (cM/Mb)',xlabel=f'{t.chrom} position (Mb, GRCh38)')


def draw_bgs(ax,bgs,t):
    d=bgs[(bgs.chrom==t.chrom)&(bgs.end>t.display_start0)&(bgs.start<t.display_end0)]
    # Horizontal segments retain the deposited 100-kb bins and masked gaps.
    for r in d.itertuples():ax.plot(np.array([r.start+1,r.end])/1e6,[r.Bprime,r.Bprime],color=DARK,lw=1.2)
    if len(d)==0:ax.text(.5,.5,'No published bins',ha='center',va='center',transform=ax.transAxes)
    ax.set(ylim=(0,1),yticks=[0,.5,1],ylabel=r"BGS $B^{\prime}$",
           xlim=((t.display_start0+1)/1e6,t.display_end0/1e6),xlabel=f'{t.chrom} position (Mb, GRCh38)')
    ax.axvspan((t.gene_start0+1)/1e6,t.gene_end0/1e6,color=PALE,zorder=0)


def check(fig):
    for ax in fig.axes:
        if not ax.axison:continue
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            ticks=axis.get_majorticklocs();axis.set_ticks(ticks[(ticks>=min(limits))&(ticks<=max(limits))])
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,t.get_text()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--case-data',type=Path,required=True)
    a=ap.parse_args();out=a.root/'results';out.mkdir(exist_ok=True)
    targets=pd.read_csv(a.root/'inputs/targets.tsv',sep='\t');complete=[]
    for t in targets.itertuples():
        if all((a.root/'maps'/f'{t.gene}.{p}.json').exists() for p in t.populations.split(',')):complete.append(t)
    summary=pd.read_csv(out/'gene_recombination_summary.tsv',sep='\t')
    bgs=pd.read_csv(a.root/'bgs/buffalo_kern2024_YRI_CADD6_100kb.tsv.gz',sep='\t')
    with PdfPages(out/'highlighted_gene_recombination_atlas.pdf') as pdf:
        for page in range(0,len(complete),6):
            fig=plt.figure(figsize=(8.3,10.2))
            grid=fig.add_gridspec(3,2,left=.095,right=.98,bottom=.055,top=.94,hspace=.52,wspace=.28)
            for slot,t in enumerate(complete[page:page+6]):
                pair=grid[slot//2,slot%2].subgridspec(2,1,height_ratios=[2,1],hspace=.13)
                ax=fig.add_subplot(pair[0]);bx=fig.add_subplot(pair[1],sharex=ax)
                draw_rate(ax,a.root,t)
                ax.set_xlabel('');ax.tick_params(labelbottom=False)
                draw_bgs(bx,bgs,t)
                ax.legend(frameon=False,fontsize=6.5,loc='lower left',bbox_to_anchor=(0,1.01),
                          ncol=3,title=t.gene,title_fontsize=8,borderaxespad=0,columnspacing=1)
            check(fig);pdf.savefig(fig);plt.close(fig)
    if all(any(t.gene==g for t in complete) for g in ['GRK2','TREM2']):
        fig,axes=plt.subplots(4,2,figsize=(7.2,7.3),sharex='col',gridspec_kw={'height_ratios':[1.1,1.1,1,.8]})
        fig.subplots_adjust(left=.10,right=.98,top=.965,bottom=.08,hspace=.23,wspace=.30)
        for col,gene in enumerate(['GRK2','TREM2']):
            t=next(t for t in complete if t.gene==gene);z=np.load(a.case_data/f'{gene}_plot_data.npz')
            for tag,pop,color,style in [('focal',t.focal_population,DARK,'-'),('control','YRI',LIGHT,'--')]:
                x=z[tag+'_positions']/1e6
                axes[0,col].fill_between(x,z[tag+'_q25'],z[tag+'_q75'],color=color,alpha=.2,lw=0)
                axes[0,col].plot(x,z[tag+'_median'],color=color,ls=style,lw=1,label=pop)
            axes[0,col].set(yscale='log',ylabel='TMRCA (generations)',)
            axes[0,col].legend(frameon=False,fontsize=7,title=gene if gene=='GRK2' else 'TREML1 / TREM2',title_fontsize=8)
            draw_rate(axes[1,col],a.root,t,split=True);axes[1,col].set_xlabel('')
            axes[2,col].plot(z['h12_positions']/1e6,z['h12'],color=DARK,lw=1)
            axes[2,col].set(ylim=(0,1),ylabel=r"Garud's $H_{12}$")
            draw_bgs(axes[3,col],bgs,t)
            for ax in axes[:,col]:ax.axvspan((t.gene_start0+1)/1e6,t.gene_end0/1e6,color=PALE,zorder=0)
            if gene=='TREM2':
                neighbour=pd.read_csv(a.case_data/'TREM2_gene_track.csv').set_index('gene_name').loc['TREML1']
                for ax in axes[:,col]:ax.axvspan(neighbour.start/1e6,neighbour.end/1e6,color=PALE,zorder=0)
            for row,ax in enumerate(axes[:,col]):ax.text(-.19,1.03,'abcdefgh'[2*row+col],transform=ax.transAxes,fontsize=10)
        check(fig)
        for ext in ['pdf','png']:fig.savefig(out/f'focal_recombination_comparison.{ext}',dpi=300)
        plt.close(fig)
    common=("fastRho v0.1.1 phased estimates, shown as overlap-weighted 10-kb means on a logarithmic axis. "
            "Absolute rates are conditional on Ne=10,000; raw population-scaled rho is retained in the map files. "
            "The grey dotted line is the original Gamma-SMC constant input (1 cM/Mb). "
            "BGS B′ is the retained neutral diversity fraction from Buffalo and Kern (2024), DOI 10.1371/journal.pgen.1011144, "
            "CADD 6%/deCODE altgrid YRI fit, in native GRCh38 100-kb bins. Lower B′ means stronger predicted BGS. "
            "Masked bins are gaps; there is no interpolation. This shared YRI reference is not a separate population-specific "
            "BGS estimate for the focal populations. Recombination and BGS are contextual annotations: BGS, demography, "
            "and positive selection, including combinations, remain alternative explanations for TMRCA patterns.")
    (out/'figure_legends.md').write_text(
        '# Focal recombination and BGS comparison\n\n'
        'GRK2 (left, GIH versus YRI) and TREML1/TREM2 (right, IBS versus YRI). '
        '(a,b) Existing Gamma-SMC median and interquartile TMRCA across 20 pairs. '
        '(c,d) fastRho recombination; faint dotted curves show disjoint diploid-sample halves. '
        '(e,f) Existing focal-population H12. (g,h) Buffalo–Kern BGS. '
        'Dark solid lines denote GIH/IBS, light dashed lines YRI. Pale shading marks GRK2 or TREML1 and TREM2. '
        +common+'\n\n# Highlighted-gene atlas\n\n'
        f'{len(complete)} genes, six loci per page in target-manifest order. Each locus has aligned recombination (upper) '
        'and BGS (lower) tracks. The legend identifies the gene and populations. Pale shading marks the named gene body. '
        +common+'\n')
    (out/'figure_validation.json').write_text(json.dumps({'atlas_genes':len(complete),'atlas_pages':(len(complete)+5)//6,'text_bounds_checked':True,'titles_and_subtitles':False,'BGS_bin_width_bp':100000},indent=2)+'\n')

if __name__=='__main__':main()
