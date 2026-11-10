"""Annotated purifying-selection fits; preserve frozen targets and replicate means."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np
from metrics import scores

# Match the manuscript's NPG-inspired population palette. Styles also identify models.
NAVY, CYAN, CORAL, TEAL = '#3C5488', '#4DBBD5', '#E64B35', '#00A087'
GREY, GUIDE, PALE = '#AEB6BF', '#69737D', '#F0F2F7'
POPULATION_COLORS = {'GIH': CYAN, 'IBS': NAVY}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_line(ax, x, y, **kwargs):
    artist, = ax.plot(x, y, **kwargs)
    np.testing.assert_array_equal(artist.get_xdata(), x)
    np.testing.assert_array_equal(artist.get_ydata(), y)
    return artist


def effective_parameters(config, meta):
    result=dict(config)
    if config['model']=='negative':
        result.update(scaled_Ne=config['Ne']/config['Q'],scaled_s=config['s']*config['Q'],
                      burn_generations=config['burn_Ne']*config['Ne'])
        if config['target']=='block':
            center=meta['center0']+config['offset'];width=config['width']
            result.update(selected_interval_1based=[center-width//2+1,center+(width-width//2)],
                          target_interpretation='Synthetic contiguous selected target, not annotated human function')
        else:
            result['target_interpretation']='Havana exons union phastCons conserved sequence'
            result['inactive_fields']=['width','offset']
    return result


def neutral_description(config):
    if 'ancient_Ne' in config:
        return (f"Fitted neutral: Nₑ = {config['Ne']:,.0f} now; "
                f"{config['ancient_Ne']:,.0f} before {config['change_time']:,.0f} generations.")
    return f"Fitted neutral: constant Nₑ = {config['Ne']:,.1f}."


def load_panel(args, key, settings, target):
    """Snapshot completed evidence once; never label fitting seeds as validation."""
    run = args.run/key
    snapshot = args.out/'input_snapshot'/key
    snapshot.mkdir(parents=True, exist_ok=True)
    history_bytes = (run/'history.json').read_bytes()
    (snapshot/'history.json').write_bytes(history_bytes)
    history = json.loads(history_bytes)
    inputs = [snapshot/'history.json']
    if (run/'summary.json').exists():
        summary = read_json(run/'summary.json')
        stage = 'fresh-seed validation'
        paths = {'best': run/'validation_best.npz',
                 'neutral': run/'validation_tuned_neutral.npz',
                 'replicate': run/'validation/best/rep_000/complete.json',
                 'summary': run/'summary.json'}
    else:
        if not args.live:
            raise FileNotFoundError(f'{run}/summary.json; use --live for provisional fitting results')
        completed = [r for r in history if r['train_loss'] is not None]
        chosen = min((r for r in completed if r['config']['model']=='negative'),
                     key=lambda r: r['train_loss'])
        neutral = min((r for r in completed if r['config']['model']=='neutral'),
                      key=lambda r: r['train_loss'])
        summary = {'chosen': {'index': chosen['candidate'], 'config': chosen['config'],
                              'train_loss': chosen['train_loss']},
                   'best_neutral': neutral['config'],
                   'training_replicates': settings['replicates']}
        stage = 'fitting seeds; provisional'
        paths = {'best': run/f"candidate_{chosen['candidate']:04d}/ensemble.npz",
                 'neutral': run/f"candidate_{neutral['candidate']:04d}/ensemble.npz",
                 'replicate': run/f"candidate_{chosen['candidate']:04d}/rep_000/complete.json"}
    source_paths = {}
    for name, source in paths.items():
        dest = snapshot/(name+source.suffix)
        shutil.copyfile(source, dest)
        source_paths[name] = str(source)
        inputs.append(dest)
    with np.load(snapshot/'best.npz') as z:
        values, truth = z['inferred'], z['truth']
    with np.load(snapshot/'neutral.npz') as z:
        neutral_values = z['inferred']
    summary['display_stage'] = stage
    summary['display_scores'] = {'best': scores(values.mean(axis=0), target),
                                 'tuned_neutral': scores(neutral_values.mean(axis=0), target)}
    if stage == 'fresh-seed validation':
        for name in summary['display_scores']:
            np.testing.assert_allclose(summary['display_scores'][name]['all_bins'],
                                       summary['validation'][name]['all_bins'], rtol=1e-12)
    else:
        np.testing.assert_allclose(summary['display_scores']['best'][settings['fit_bins']],
                                   summary['chosen']['train_loss'], rtol=1e-12)
    return summary, history, values, truth, neutral_values, read_json(snapshot/'replicate.json'), inputs, source_paths


def parameter_card(ax,summary,meta,effective):
    ax.axis('off');c=summary['chosen']['config']
    ax.text(0,1.04,f"Retained #{summary['chosen']['index']} · negative selection",fontsize=9.3,va='top',transform=ax.transAxes)
    rows=[('Generating Nₑ',f"{c['Ne']:,.1f}".removesuffix('.0')),
          ('Homozygous s',f"{c['s']:.5g}".replace('-','−')),
          ('h / deleterious fraction',f"{c['h']:.2g} / {c['selected_fraction']:.3g}")]
    if c['target']=='block':
        lo,hi=effective['selected_interval_1based']
        rows += [('Synthetic target',f"{lo/1e6:.4f}–{hi/1e6:.4f} Mb"),
                 ('Width / centre offset',f"{c['width']/1000:g} / {c['offset']/1000:+.1f} kb".replace('-','−'))]
    else:rows += [('Selected target','Exons + conserved sequence'),
                  ('Selected bases',f"{effective['selected_bp']:,} bp")]
    rows += [('Burn-in',f"{c['burn_Ne']:g} Nₑ = {c['burn_Ne']*c['Ne']:,.0f} gen"),
             ('Q / simulated diploids',f"{c['Q']} / {c['Ne']/c['Q']:,.0f}"),
             ('Haplotypes / pairs',f"{meta['n_haplotypes']} / {meta['pairs']}"),
             ('Inference Nₑ',f"{meta['calibration']['calibrated_Ne']:,.1f}")]
    spacing=.077
    for j,(label,value) in enumerate(rows):
        y=.88-spacing*j
        ax.text(0,y,label,fontsize=8.0,va='center',transform=ax.transAxes)
        ax.text(1,y,value,fontsize=8.0,va='center',ha='right',transform=ax.transAxes)
    y=.88-spacing*len(rows)-.025
    score=summary['display_scores']['best']['all_bins']
    ax.plot([0,1],[y+.04]*2,color='#DCE1E6',lw=.7,transform=ax.transAxes)
    score_label = 'Validation log error' if summary['display_stage']=='fresh-seed validation' else 'Fitting log error'
    ax.text(0,y,score_label,fontsize=8.4,va='center',transform=ax.transAxes)
    ax.text(1,y,f'{score:.3f}',fontsize=10,va='center',ha='right',transform=ax.transAxes)
    improvement=100*(1-score/summary['display_scores']['tuned_neutral']['all_bins'])
    word='lower' if improvement>=0 else 'higher'
    ax.text(1,y-.085,f'{abs(improvement):.1f}% {word} than fitted neutral',fontsize=8.0,va='center',ha='right',transform=ax.transAxes)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['targets', 'run', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--positive-run', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Use completed fitting ensembles where validation is unfinished')
    parser.add_argument('--fonts', type=Path, help='Optional directory of manuscript TrueType fonts')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fonts:
        for path in sorted(args.fonts.glob('*.ttf')):
            font_manager.fontManager.addfont(path)
    manifest = read_json(args.run/'manifest.json')
    keys = manifest['settings']['keys']
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    fig = plt.figure(figsize=(6.5*len(keys), 8.4))
    fig.text(.065, .978, 'Autoresearch · negative-selection TMRCA profile matching', fontsize=14, va='top')
    incomplete = any(not (args.run/key/'summary.json').exists() for key in keys)
    subtitle = ('Figure draft · search in progress · evaluation stage shown for each locus' if incomplete else
                'Recurrent deleterious mutations · frozen targets and scoring rule · fresh-seed validation')
    fig.text(.065, .947, subtitle,
             fontsize=9, va='top')
    handles = [
        Line2D([], [], color=CYAN, lw=1.7, label='Observed GIH'),
        Line2D([], [], color=NAVY, lw=1.7, label='Observed IBS'),
        Line2D([], [], color=CORAL, lw=1.5, label='Negative selection: Gamma'),
        Patch(facecolor=CORAL, alpha=.14, label='10–90% replicate spread'),
        Line2D([], [], color=CORAL, lw=1, ls='--', label='Negative selection: true TMRCA'),
        Line2D([], [], color=TEAL, lw=1, ls='-.', label='Best tested neutral'),
        Line2D([], [], color=GUIDE, lw=1, ls=':', label='Previous positive-selection fit'),
        Patch(facecolor=CORAL, alpha=.05, edgecolor=CORAL, label='Synthetic selected target'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.061, .917), ncol=4,
               frameon=False, handlelength=2.8, columnspacing=3.0, labelspacing=.6)
    reports, axes, lower_values, upper_values = {}, [], [], []
    for i, key in enumerate(keys):
        left = .065 + i*.495
        width = .405
        target_path = args.targets/key/'target.npz'
        meta_path = args.targets/key/'manifest.json'
        meta = read_json(meta_path)
        with np.load(target_path) as z:
            target = {k: z[k] for k in z.files}
        summary, history, values, truth, neutral_values, replicate_meta, inputs, source_paths = load_panel(args, key, manifest['settings'], target)
        neutral = neutral_values.mean(axis=0)
        x = (target['edges'][:-1] + target['edges'][1:])/2e6
        observed, inferred, true_mean = target['observed'], values.mean(axis=0), truth.mean(axis=0)
        low, high = np.quantile(values, [.1, .9], axis=0)
        effective = effective_parameters(summary['chosen']['config'], meta)
        effective['selected_bp']=replicate_meta['selected_bp']
        ax = fig.add_axes([left, .525, width, .305])
        axes.append(ax)
        ax.axvspan(*np.array(meta['gene_interval0'])/1e6, color=PALE, ec='none', zorder=0)
        ax.fill_between(x, low, high, color=CORAL, alpha=.14, lw=0, zorder=1)
        with np.load(args.positive_run/key/'validation_best.npz') as z:
            positive=z['inferred'].mean(axis=0)
        checked_line(ax,x,positive,color=GUIDE,ls=':',lw=1.1,zorder=2)
        inputs += [target_path, meta_path, args.positive_run/key/'validation_best.npz']
        checked_line(ax, x, neutral, color=TEAL, ls='-.', lw=1, zorder=2)
        checked_line(ax, x, true_mean, color=CORAL, ls='--', lw=1, zorder=3)
        checked_line(ax, x, inferred, color=CORAL, lw=1.5, zorder=4)
        checked_line(ax, x, observed, color=POPULATION_COLORS.get(meta['population'], NAVY),
                     lw=1.8, zorder=5)
        ax.set(yscale='log', xlim=target['edges'][[0, -1]]/1e6,
               ylabel='Mean pairwise TMRCA (generations)',
               xlabel=f"Chromosome {meta['chromosome']} position (Mb, GRCh38)")
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.arange(np.ceil(xmin*5), np.floor(xmax*5)+1)/5)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(NullLocator())
        # Keep the actual gene distinct from a fitted synthetic selected interval.
        ax.text(np.mean(meta['gene_interval0'])/1e6+.013, .96, meta['gene'],
                transform=ax.get_xaxis_transform(), fontstyle='italic', fontsize=9,
                ha='left', va='top')
        if 'selected_interval_1based' in effective:
            lo,hi=effective['selected_interval_1based']
            ax.axvspan((lo-1)/1e6,hi/1e6,facecolor=CORAL,alpha=.045,ec='none',zorder=0)
            ax.plot([(lo-1)/1e6,hi/1e6],[.985,.985],transform=ax.get_xaxis_transform(),color=CORAL,lw=1.2)
        fig.text(left-.028, .853, chr(97+i), fontsize=12)
        stage_label = f'{len(values)} fresh validation replicates' if summary['display_stage']=='fresh-seed validation' else f'{len(values)} fitting replicates · provisional'
        fig.text(left, .853, f"{meta['gene']} · {meta['population']}   |   {stage_label}", fontsize=10)
        lower_values.extend([observed.min(), low.min(), true_mean.min(), neutral.min(), positive.min()])
        upper_values.extend([observed.max(), high.max(), true_mean.max(), neutral.max(), positive.max()])

        valid = [row for row in history if row['train_loss'] is not None and row['config']['model']=='negative']
        steps = np.array([row['candidate'] for row in valid])
        losses = np.array([row['train_loss'] for row in valid])
        running = np.minimum.accumulate(losses)
        trace = fig.add_axes([left, .239, .164, .174])
        trace.scatter(steps, losses, c=GREY, s=22, zorder=3, edgecolors='white', linewidths=.4)
        trace.step(steps, running, where='post', color=CORAL, lw=1.3, zorder=2)
        keeps = np.array([row['status'] == 'keep' for row in valid])
        trace.scatter(steps[keeps], losses[keeps], c=CORAL, marker='s', s=19, zorder=4)
        chosen = summary['chosen']
        trace.scatter([chosen['index']], [chosen['train_loss']], c=CORAL, edgecolor='black',
                      linewidth=.5, marker='*', s=90, zorder=5)
        trace.annotate(f"#{chosen['index']}", (chosen['index'], chosen['train_loss']),
                       xytext=(5, 9), textcoords='offset points', fontsize=8)
        trace.set(xlabel='Negative-selection candidate', ylabel='Training log error',
                  xlim=(min(steps)-.4, max(steps)+.4), xticks=steps[::4])
        trace.margins(y=.17)
        fig.text(left-.028, .447, chr(97+len(keys)+i), fontsize=12)
        fig.text(left, .447, f"Search · {len(valid)} candidates × {summary['training_replicates']} seeds", fontsize=9.4)
        trace.legend(handles=[Line2D([], [], color=CORAL, marker='s', ms=3, lw=1,
                                    label='Retained'),
                              Line2D([], [], color=GREY, marker='o', ms=4, lw=0,
                                    label='Tested')],
                     loc='upper left', bbox_to_anchor=(0, 1.13), ncol=2,
                     frameon=False, fontsize=7.3, columnspacing=.9,
                     handlelength=1.1, handletextpad=.4, borderaxespad=.15)
        card = fig.add_axes([left+.195, .180, .230, .242])
        rows = parameter_card(card, summary, meta, effective)
        if 'best_neutral' in summary:
            fig.text(left, .163, neutral_description(summary['best_neutral']), fontsize=8.2)
        reports[key] = {
            'source_sha256': {str(p): sha256(p) for p in inputs},
            'effective_simulation_parameters': effective, 'displayed_parameter_rows': rows,
            'inference_calibration': meta['calibration'], 'evaluation_scores': summary['display_scores'],
            'evaluation_stage': summary['display_stage'], 'displayed_replicates': len(values),
            'neutral_replicates': len(neutral_values), 'positive_reference_replicates': 24,
            'retained_candidate': summary['chosen'], 'completed_negative_candidates': len(valid),
            'training_replicates': summary['training_replicates'], 'source_paths': source_paths, 'bins': len(x),
            'x_coordinate_convention': 'Bin midpoint in zero-based Mb; selected-interval parameter card uses one-based endpoints',
            'line_arrays_equal_to_source': True,
        }
    ymin = min(700., min(lower_values)*.75)
    ymax = max(130000., max(upper_values)*1.15)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([10.**p for p in range(-1, 9) if ymin <= 10.**p <= ymax])
    # Shared settings are checked before placing a common caption on the figure.
    calibrations = [r['inference_calibration'] for r in reports.values()]
    assert all(c['physical_mu'] == 1.25e-8 and c['physical_rho'] == 1e-8 for c in calibrations)
    fig.text(.065,.118,r'Simulation: ongoing purifying selection, fixed negative DFE; μ = 1.25 × $10^{-8}$ /bp/generation; deCODE GRCh38 map; 3 Mb simulated.',fontsize=8.5)
    fig.text(.065,.091,r'Profiles: 960 kb / 20-kb bins; 20 pairs. Gamma uses actual polymorphisms, fixed chromosome Nₑ and ρ = $10^{-8}$ /bp/gen. Prior positive fit: 24 fresh seeds.',fontsize=8.5)
    fig.text(.065,.064,'Nₑ is diploid; genotype fitnesses are 1, 1 + hs, 1 + s. Q rescales population size, rates and time approximately; physical parameters are shown.',fontsize=8.5)
    fig.text(.065,.037,'Synthetic targets are fitted model regions, not human annotations. All bins fitted; new seeds validate. Settings are exploratory, not biological estimates.',fontsize=8.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= -1 and bounds.x1 <= fig.bbox.width+1 and bounds.y0 >= -1 and bounds.y1 <= fig.bbox.height+1, (artist.get_text(), bounds)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.out/f'profile_matches.{ext}', dpi=300)
    plt.close(fig)
    report = {
        'presentation_only': True, 'model_family': 'recurrent negative selection', 'run_manifest_sha256': sha256(args.run/'manifest.json'),
        'plot_source_sha256': sha256(Path(__file__)), 'yscale': 'log',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'draft': incomplete,
        'statistic': 'Arithmetic mean across the frozen 20 pairs; mean across simulation replicates at the explicitly labelled evaluation stage',
        'ribbon': '10th–90th percentile across simulation replicates; not pairwise IQR or parameter uncertainty',
        'panels': reports, 'outputs': {str(p.name): sha256(p) for p in args.out.glob('profile_matches.*')},
    }
    (args.out/'figure_metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.out), 'loci': keys, 'array_checks': 'passed'}))


if __name__ == '__main__':
    main()
