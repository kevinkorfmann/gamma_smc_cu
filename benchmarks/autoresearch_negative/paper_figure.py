"""Caption-led, paper-sized 2x2 model comparison and separate 2x2 SI parameters."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
import numpy as np

from metrics import scores
from plot import load_panel, effective_parameters, checked_line

CYAN, NAVY, CORAL, TEAL = '#4DBBD5', '#3C5488', '#E64B35', '#00A087'
GREY, PALE = '#AEB6BF', '#F0F2F7'
YLIM = (350, 250000)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep_panel(args, key, target):
    src = args.positive_run/key
    dest = args.out/'input_snapshot'/'sweep'/key
    dest.mkdir(parents=True)
    names = ['summary.json', 'history.json', 'validation_best.npz', 'validation_tuned_neutral.npz']
    for name in names:
        shutil.copyfile(src/name, dest/name)
    s, history = read(dest/'summary.json'), read(dest/'history.json')
    with np.load(dest/'validation_best.npz') as z:
        values, truth = z['inferred'], z['truth']
    with np.load(dest/'validation_tuned_neutral.npz') as z:
        neutral = z['inferred']
    s['display_stage'] = 'fresh-seed validation'
    s['display_scores'] = {label: scores(arr.mean(axis=0), target)
                           for label, arr in [('best', values), ('tuned_neutral', neutral)]}
    for label in s['display_scores']:
        np.testing.assert_allclose(s['display_scores'][label]['all_bins'],
                                   s['validation'][label]['all_bins'], rtol=1e-12)
    return s, history, values, truth, neutral, [dest/name for name in names]


def parameter_rows(panel):
    s, c, m = panel['summary'], panel['summary']['chosen']['config'], panel['meta']
    e = panel['effective']
    rows = [('Model', 'Hard sweep' if panel['family']=='sweep' else 'Negative selection'),
            ('Retained candidate', f"#{s['chosen']['index']}"),
            ('Generating Nₑ', f"{c['Ne']:,.2f}".rstrip('0').rstrip('.'))]
    if panel['family']=='sweep':
        rows += [('Genic s', f"{c['s']:.6g}"),
                 ('Since sweep endpoint', f"{c['age']:,.3f}".rstrip('0').rstrip('.')+' gen'),
                 ('Selected site (GRCh38)', f"{e['selected_position_1based']:,} bp"),
                 ('Centre offset', f"{c['offset']/1000:+.3f} kb"),
                 ('Start → end frequency', f"{e['effective_start_frequency']:.6f} → {e['effective_end_frequency']:.6f}"),
                 ('Simulation engine', 'msprime; conditioned sweep')]
    else:
        rows += [('Homozygous s / h', f"{c['s']:.6g} / {c['h']:g}"),
                 ('Deleterious fraction', f"{c['selected_fraction']:.4g}")]
        if c['target']=='block':
            rows += [('Synthetic target width', f"{c['width']/1000:g} kb"),
                     ('Centre offset', f"{c['offset']/1000:+.3f} kb")]
        else:
            rows += [('Selected target', 'Exons + conserved sequence'),
                     ('Selected bases', f"{e['selected_bp']:,} bp")]
        rows += [('Q / simulated diploids', f"{c['Q']:g} / {c['Ne']/c['Q']:,.0f}"),
                 ('Forward burn-in', f"{c['burn_Ne']:g} Nₑ = {c['burn_Ne']*c['Ne']:,.0f} gen")]
    rows += [('Haplotypes / pairs', f"{m['n_haplotypes']} / {m['pairs']}"),
             ('Inference Nₑ', f"{m['calibration']['calibrated_Ne']:,.2f}"),
             ('Fitting / displayed replicates', f"{s['training_replicates']} / {len(panel['values'])}"),
             ('Displayed ensemble', 'Fresh validation' if s['display_stage']=='fresh-seed validation' else 'Fitting; provisional'),
             ('Model / neutral log error', f"{s['display_scores']['best']['all_bins']:.3f} / {s['display_scores']['tuned_neutral']['all_bins']:.3f}")]
    return [(a,b.replace('-', '−')) for a,b in rows]


def export(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for t in visible:
        assert to_rgba(t.get_color()) == (0.,0.,0.,1.), ('Non-black text',t.get_text(),t.get_color())
        b = t.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, (t.get_text(), b)
    assert min(t.get_fontsize() for t in visible) >= 6
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=600)
    # Independent grayscale check, leaving publication colours untouched.
    from PIL import Image
    with Image.open(stem.with_suffix('.png')) as im:
        im.convert('L').save(stem.with_name(stem.name+'_grayscale').with_suffix('.png'))
    plt.close(fig)


def captions(panels):
    stages = '; '.join(f"{p['letter']}: {len(p['values'])} "+
                      ('fresh-seed validation replicates' if p['summary']['display_stage']=='fresh-seed validation'
                       else 'fitting replicates, provisional') for p in panels)
    main = (
        '**Selection models fitted to genomic TMRCA profiles.** '
        '(a,b) Conditioned hard-sweep simulations; (c,d) recurrent negative-selection simulations. '
        'Left: GRK2–GIH; right: TREM2–IBS. Cyan/navy solid lines show the observed profiles; '
        'coral dashed lines show inferred Gamma-SMC profiles from the retained simulation model, '
        'and coral dotted lines show its true genealogical TMRCA. Teal dash-dotted lines show the best tested neutral model. '
        'Coral ribbons span the 10th–90th percentiles across simulation replicates, not confidence intervals. '
        'Grey strips mark the focal gene; coral triangles mark sweep sites, and pale coral intervals with upper bars '
        'mark fitted synthetic negative-selection targets. Profiles are arithmetic means across 20 pairs, '
        'averaged in physical 20-kb bins; simulated curves also average across replicates. '
        'All panels use the preceding focal-locus figure’s log scale of 350–250,000 generations. '
        f'Ensembles: {stages}. '
        'Each simulation covers 3 Mb; the central 960 kb is shown and fitted. Model selection uses the fixed '
        'weighted log-error objective; fresh seeds assess reproducibility with no held-out genomic bins. '
        'Sweep inference uses neutral markers; negative-selection inference uses actual selected and neutral polymorphisms. '
        'The comparison is descriptive and does not identify the biological cause of the empirical troughs. '
        'Simulation parameters, search traces and approximate SLiM rescaling are given in the companion SI figure.'
    )
    si = (
        '**Search trajectories and retained simulation parameters.** '
        'Panels correspond to the main model-comparison figure: (a,b) GRK2 and TREM2 sweeps; '
        '(c,d) GRK2 and TREM2 negative selection. Grey points show completed selected-model candidates, '
        'the coral staircase gives the running minimum fitting error, squares mark improvements, '
        'and stars identify the retained candidate. Candidate numbers retain their original indices; neutral controls '
        'are excluded from these traces but included in the reported neutral comparison. '
        'The sweep trace is the final nine-configuration refinement campaign; earlier screening is preserved in the run archive. '
        'The objective is the square root of 0.75 times mean squared log error across all bins plus 0.25 times '
        'mean squared log error in the central 200 kb. All reported parameters are in physical units. '
        'Nₑ is diploid. Sweep age is time since the conditioned endpoint, not allele origin. '
        'Negative-selection fitnesses are 1, 1+hs and 1+s; the deleterious fraction applies inside selected targets. '
        'For SLiM, Q divides census size and approximately rescales rates, selection and time; Q=10 is not an unscaled equivalence claim. '
        'Common settings are mutation rate 1.25×10⁻⁸/bp/generation, deCODE GRCh38 recombination, 3-Mb simulation context, '
        'and Gamma-SMC inference with fixed chromosome calibration and scalar recombination rate 10⁻⁸/bp/generation. '
        'Synthetic blocks are fitted model regions, not human functional annotations. '
        'The fitted neutral GRK2 model has constant Nₑ=3,341.6984; the TREM2 neutral model has Nₑ=3,000 '
        'currently and Nₑ=25,000 before 3,000 generations. '
        f'Ensembles: {stages}. Parameters are exploratory fits, not biological estimates.'
    )
    return main, si


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['targets','positive-run','negative-run','out','paper-style-reference']:
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--fonts', type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.fonts:
        for path in args.fonts.glob('*.ttf'):
            font_manager.fontManager.addfont(path)
    # Exact scale and physical text width of gen_fig_focal_comparison.py.
    assert 'set_ylim(350,250000)' in args.paper_style_reference.read_text()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.2,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.6,'ytick.major.size':2.6,
        'mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.facecolor':'white','axes.facecolor':'white'})
    panels = []
    negative_manifest = read(args.negative_run/'manifest.json')
    for row, family in enumerate(['sweep','negative']):
        for col,key in enumerate(['GRK2_GIH','TREM2_IBS']):
            meta = read(args.targets/key/'manifest.json')
            with np.load(args.targets/key/'target.npz') as z:
                target = {k:z[k] for k in z.files}
            if family=='sweep':
                s, history, values, truth, neutral, inputs = sweep_panel(args,key,target)
                c = s['chosen']['config']
                effective = dict(c, effective_start_frequency=1/(2*c['Ne']),
                    effective_end_frequency=min(c['end_frequency'],1-1/(2*c['Ne'])),
                    selected_position_1based=meta['center0']+c['offset']+1)
            else:
                opts = SimpleNamespace(run=args.negative_run, out=args.out/'negative', live=True)
                s, history, values, truth, neutral, rep, inputs, _ = load_panel(opts,key,negative_manifest['settings'],target)
                effective = effective_parameters(s['chosen']['config'],meta)
                effective['selected_bp'] = rep['selected_bp']
            inputs += [args.targets/key/'manifest.json',args.targets/key/'target.npz']
            panels.append(dict(letter='abcd'[row*2+col],row=row,col=col,key=key,family=family,
                meta=meta,target=target,summary=s,history=history,values=values,truth=truth,
                neutral=neutral,effective=effective,inputs=inputs))
    fig = plt.figure(figsize=(6.5,4.15))
    reports = {}
    for p in panels:
        x0, y0 = .49+p['col']*3.23, .45+(1-p['row'])*1.89
        ax = fig.add_axes([x0/6.5,y0/4.15,2.69/6.5,1.54/4.15])
        target, meta = p['target'], p['meta']
        x = (target['edges'][:-1]+target['edges'][1:])/2e6
        mean, truth, neutral = p['values'].mean(0),p['truth'].mean(0),p['neutral'].mean(0)
        lo, hi = np.quantile(p['values'],[.1,.9],axis=0)
        ax.axvspan(*np.asarray(meta['gene_interval0'])/1e6,color=PALE,ec='none',zorder=0)
        ax.fill_between(x,lo,hi,color=CORAL,alpha=.14,lw=0,zorder=1)
        for arr,kwargs in [(neutral,dict(color=TEAL,ls='-.',lw=.7,zorder=2)),
                           (truth,dict(color=CORAL,ls=(0,(1,2)),lw=.8,zorder=3)),
                           (mean,dict(color=CORAL,ls=(0,(4,1.7)),lw=1,zorder=4)),
                           (target['observed'],dict(color=CYAN if p['col']==0 else NAVY,lw=1.05,zorder=5))]:
            checked_line(ax,x,arr,**kwargs)
            assert np.all((arr>=YLIM[0]) & (arr<=YLIM[1])), (p['letter'],arr.min(),arr.max())
        assert lo.min()>=YLIM[0] and hi.max()<=YLIM[1], 'Replicate ribbon would be clipped'
        if p['family']=='sweep':
            ax.plot([(p['effective']['selected_position_1based']-1)/1e6],[.982],
                    marker='v',ms=3.5,color=CORAL,transform=ax.get_xaxis_transform(),clip_on=False)
        elif 'selected_interval_1based' in p['effective']:
            a,b=p['effective']['selected_interval_1based']
            ax.axvspan((a-1)/1e6,b/1e6,color=CORAL,alpha=.045,lw=0,zorder=0)
            ax.plot([(a-1)/1e6,b/1e6],[.982,.982],color=CORAL,lw=.8,transform=ax.get_xaxis_transform())
        ax.set(yscale='log',ylim=YLIM,xlim=target['edges'][[0,-1]]/1e6)
        ax.set_yticks([1e3,1e4,1e5]);ax.yaxis.set_minor_locator(NullLocator())
        xmin,xmax=ax.get_xlim();ax.set_xticks(np.arange(np.ceil(xmin*5),np.floor(xmax*5)+1)/5)
        if p['col']==0: ax.set_ylabel('Mean TMRCA (gen.)',labelpad=2)
        if p['row']==1: ax.set_xlabel(f"Chr {meta['chromosome']} position (Mb, GRCh38)",labelpad=3)
        ax.text(0,1.045,p['letter'],transform=ax.transAxes,fontsize=8.2,va='bottom')
        ax.text(.018,.945,f"{meta['gene']} · {meta['population']}",
                transform=ax.transAxes,fontsize=6.7,va='top',zorder=8)
        model_label = 'Sweep' if p['family']=='sweep' else 'Negative selection'
        if p['summary']['display_stage']!='fresh-seed validation':
            model_label += ' · provisional'
        ax.text(.018,.855,model_label,transform=ax.transAxes,fontsize=6.2,
                color='black',va='top',zorder=8)
        reports[p['letter']]={'locus':p['key'],'family':p['family'],
            'inline_annotations':[f"{meta['gene']} · {meta['population']}",model_label],
            'retained_candidate':p['summary']['chosen'],'effective_parameters':p['effective'],
            'evaluation_stage':p['summary']['display_stage'],'replicates':len(p['values']),
            'evaluation_scores':p['summary']['display_scores'],'parameter_rows':parameter_rows(p),
            'source_sha256':{str(f):sha(f) for f in p['inputs']},
            'plotted_arrays_equal_source':True,'no_lines_or_ribbons_clipped':True}
    export(fig,args.out/'fig_selection_models')
    si = plt.figure(figsize=(6.5,6.55))
    for p in panels:
        x0=.49+p['col']*3.23; y0=.34+(1-p['row'])*3.20
        trace=si.add_axes([x0/6.5,(y0+2.01)/6.55,2.69/6.5,.73/6.55])
        valid=[r for r in p['history'] if r['train_loss'] is not None and r['config']['model']!='neutral']
        steps=np.array([r['candidate'] for r in valid]);losses=np.array([r['train_loss'] for r in valid])
        running=np.minimum.accumulate(losses);keeps=np.r_[True,np.diff(running)<0]
        trace.scatter(steps,losses,s=9,color=GREY,edgecolor='white',lw=.3,zorder=3)
        trace.step(steps,running,where='post',color=CORAL,lw=.8,zorder=2)
        trace.scatter(steps[keeps],losses[keeps],s=9,marker='s',color=CORAL,zorder=4)
        chosen=p['summary']['chosen']
        trace.scatter([chosen['index']],[chosen['train_loss']],s=35,marker='*',color=CORAL,edgecolor='black',lw=.35,zorder=5)
        trace.set(xlabel='Candidate',ylabel='Fitting log error',xlim=(steps.min()-.4,steps.max()+.4))
        trace.set_xticks(steps[::max(1,len(steps)//5)]);trace.margins(y=.22)
        trace.xaxis.labelpad=1;trace.yaxis.labelpad=2
        trace.text(0,1.085,p['letter'],transform=trace.transAxes,fontsize=8.2,va='bottom')
        table=si.add_axes([x0/6.5,y0/6.55,2.69/6.5,1.66/6.55]);table.axis('off')
        rows=parameter_rows(p)
        for j,(label,value) in enumerate(rows):
            y=1-j/(len(rows)-1)
            table.text(0,y,label,fontsize=6.2,va='center',transform=table.transAxes)
            table.text(1,y,value,fontsize=6.2,va='center',ha='right',transform=table.transAxes)
        reports[p['letter']]['completed_selected_candidates']=len(valid)
    export(si,args.out/'fig_si_selection_parameters')
    main_caption,si_caption=captions(panels)
    (args.out/'CAPTIONS.md').write_text(main_caption+'\n\n'+si_caption+'\n')
    metadata={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'paper_style_reference':str(args.paper_style_reference),'paper_style_sha256':sha(args.paper_style_reference),
        'source_sha256':sha(Path(__file__)),'shared_TMRCA_limits':list(YLIM),
        'main_size_inches':[6.5,4.15],'SI_size_inches':[6.5,6.55],
        'panels':reports,'draft':any(p['summary']['display_stage']!='fresh-seed validation' for p in panels),
        'outputs':{f.name:sha(f) for f in args.out.glob('fig_*.*')}}
    (args.out/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source=args.out/'render_source';source.mkdir()
    for name in ['paper_figure.py','plot.py','metrics.py']:
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    print(json.dumps({'out':str(args.out),'draft':metadata['draft'],'checks':'passed'}))


if __name__=='__main__':
    main()
