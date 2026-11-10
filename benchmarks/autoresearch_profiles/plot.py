"""Paper-styled profiles and annotated autoresearch traces from frozen run outputs."""
import argparse
import hashlib
import json
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
    """Mirror the frozen simulator, including its hard-sweep frequency convention."""
    result = dict(config)
    if config['model'] != 'neutral':
        sweep_ne = (config['ancient_Ne'] if 'ancient_Ne' in config
                    and config['age'] >= config['change_time'] else config['Ne'])
        result.update(
            effective_start_frequency=(1 / (2 * sweep_ne) if config['model'] == 'hard'
                                       else config['start_frequency']),
            effective_end_frequency=min(config['end_frequency'], 1 - 1 / (2 * sweep_ne)),
            selected_position_1based=int(meta['center0'] + config['offset'] + 1),
            sweep_Ne=sweep_ne,
            age_definition='Generations since conditioned sweep endpoint, not allele origin age',
        )
    return result


def neutral_description(config):
    if 'ancient_Ne' in config:
        return (f"Fitted neutral: Nₑ = {config['Ne']:,.0f} now; "
                f"{config['ancient_Ne']:,.0f} before {config['change_time']:,.0f} generations.")
    return f"Fitted neutral: constant Nₑ = {config['Ne']:,.1f}."


def parameter_card(ax, summary, meta, effective):
    ax.axis('off')
    config = summary['chosen']['config']
    model = {'hard': 'hard sweep', 'standing': 'standing sweep', 'neutral': 'neutral'}[config['model']]
    ax.text(0, 1.02, f"Retained #{summary['chosen']['index']} · {model}", fontsize=9.4,
            va='top', transform=ax.transAxes)
    rows = [('Generating Nₑ', f"{config['Ne']:,.1f}".removesuffix('.0'))]
    if config['model'] != 'neutral':
        rows += [
            ('Genic s', f"{config['s']:.6g}"),
            ('Since sweep end', f"{config['age']:,.3f}".rstrip('0').rstrip('.') + ' gen'),
            ('Selected site (bp)', f"{effective['selected_position_1based']:,}"),
            ('Offset from centre', f"{config['offset']/1000:+.2f} kb".replace('-', '−')),
            ('Start → end freq.', f"{effective['effective_start_frequency']:.6f} → "
                                  f"{effective['effective_end_frequency']:.6f}"),
        ]
    if 'ancient_Ne' in config:
        rows += [('Ancient Nₑ', f"{config['ancient_Ne']:,.0f}"),
                 ('Size-change time', f"{config['change_time']:,.0f} gen")]
    rows += [('Haplotypes / pairs', f"{meta['n_haplotypes']} / {meta['pairs']}"),
             ('Inference Nₑ', f"{meta['calibration']['calibrated_Ne']:,.1f}")]
    spacing = .089 if len(rows) <= 8 else .075
    for j, (label, value) in enumerate(rows):
        y = .865 - spacing*j
        ax.text(0, y, label, fontsize=8.4, va='center', transform=ax.transAxes)
        ax.text(1, y, value, fontsize=8.4, va='center', ha='right', transform=ax.transAxes)
    loss = summary['validation']['best']['all_bins']
    score_y = .865 - spacing*len(rows) - .035
    ax.plot([0, 1], [score_y+.042]*2, color='#DCE1E6', lw=.7, transform=ax.transAxes)
    ax.text(0, score_y, 'Fresh-seed log error', fontsize=8.4, va='center', transform=ax.transAxes)
    ax.text(1, score_y, f'{loss:.3f}', fontsize=10, va='center', ha='right', transform=ax.transAxes)
    if 'tuned_neutral' in summary['validation']:
        improvement = 100*(1-loss/summary['validation']['tuned_neutral']['all_bins'])
        ax.text(1, score_y-.091, f'{improvement:.1f}% lower than fitted neutral', fontsize=8.1,
                va='center', ha='right', transform=ax.transAxes)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['targets', 'run', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
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
    fig.text(.065, .978, 'Autoresearch · genomic TMRCA profile matching', fontsize=14, va='top')
    fig.text(.065, .947, 'Frozen empirical targets · conditioned simulations · retain the lowest training error',
             fontsize=9, va='top')
    handles = [
        Line2D([], [], color=CYAN, lw=1.7, label='Observed GIH'),
        Line2D([], [], color=NAVY, lw=1.7, label='Observed IBS'),
        Line2D([], [], color=CORAL, lw=1.5, label='Best sweep: Gamma mean'),
        Patch(facecolor=CORAL, alpha=.14, label='10–90% replicate spread'),
        Line2D([], [], color=CORAL, lw=1, ls='--', label='Best sweep: true mean'),
        Line2D([], [], color=TEAL, lw=1, ls='-.', label='Best tested neutral'),
        Line2D([], [], color=GREY, lw=1, ls=':', label='Fixed neutral baseline'),
        Line2D([], [], color=CORAL, marker='v', ms=5, lw=0, label='Simulated selected site'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.061, .917), ncol=4,
               frameon=False, handlelength=2.8, columnspacing=3.0, labelspacing=.6)
    reports, axes, lower_values, upper_values = {}, [], [], []
    for i, key in enumerate(keys):
        left = .065 + i*.495
        width = .405
        target_path = args.targets/key/'target.npz'
        meta_path = args.targets/key/'manifest.json'
        summary_path = args.run/key/'summary.json'
        history_path = args.run/key/'history.json'
        meta, summary = read_json(meta_path), read_json(summary_path)
        with np.load(target_path) as z:
            target = {k: z[k] for k in z.files}
        with np.load(args.run/key/'validation_best.npz') as z:
            values, truth = z['inferred'], z['truth']
        with np.load(args.run/key/'validation_neutral.npz') as z:
            neutral = z['inferred'].mean(axis=0)
        x = (target['edges'][:-1] + target['edges'][1:])/2e6
        observed, inferred, true_mean = target['observed'], values.mean(axis=0), truth.mean(axis=0)
        low, high = np.quantile(values, [.1, .9], axis=0)
        effective = effective_parameters(summary['chosen']['config'], meta)
        ax = fig.add_axes([left, .525, width, .305])
        axes.append(ax)
        ax.axvspan(*np.array(meta['gene_interval0'])/1e6, color=PALE, ec='none', zorder=0)
        ax.fill_between(x, low, high, color=CORAL, alpha=.14, lw=0, zorder=1)
        checked_line(ax, x, neutral, color=GREY, ls=':', lw=1, zorder=2)
        tuned_path = args.run/key/'validation_tuned_neutral.npz'
        inputs = [target_path, meta_path, summary_path, history_path,
                  args.run/key/'validation_best.npz', args.run/key/'validation_neutral.npz']
        if tuned_path.exists():
            with np.load(tuned_path) as z:
                tuned = z['inferred'].mean(axis=0)
            checked_line(ax, x, tuned, color=TEAL, ls='-.', lw=1, zorder=2)
            inputs.append(tuned_path)
            lower_values.append(tuned.min()); upper_values.append(tuned.max())
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
        # A gene label and the selected-site marker deliberately remain distinct.
        ax.text(np.mean(meta['gene_interval0'])/1e6+.013, .96, meta['gene'],
                transform=ax.get_xaxis_transform(), fontstyle='italic', fontsize=9,
                ha='left', va='top')
        if effective.get('selected_position_1based') is not None:
            site = (effective['selected_position_1based']-1)/1e6
            ax.plot([site], [.985], marker='v', ms=5, color=CORAL,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        fig.text(left-.028, .853, chr(97+i), fontsize=12)
        fig.text(left, .853, f"{meta['gene']} · {meta['population']}", fontsize=11)
        lower_values.extend([observed.min(), low.min(), true_mean.min(), neutral.min()])
        upper_values.extend([observed.max(), high.max(), true_mean.max(), neutral.max()])

        history = read_json(history_path)
        valid = [row for row in history if row['train_loss'] is not None]
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
        trace.set(xlabel='Candidate (final refinement)', ylabel='Training log error',
                  xlim=(-.4, max(steps)+.4), xticks=steps[::2])
        trace.margins(y=.17)
        fig.text(left-.028, .447, chr(97+len(keys)+i), fontsize=12)
        fig.text(left, .447, f"Search · {len(valid)} candidates × {summary['training_replicates']} seeds", fontsize=9.4)
        trace.legend(handles=[Line2D([], [], color=CORAL, marker='s', ms=3, lw=1,
                                    label='Best retained'),
                              Line2D([], [], color=GREY, marker='o', ms=4, lw=0,
                                    label='Other candidate')],
                     loc='upper right', frameon=False, fontsize=7.3,
                     handlelength=1.1, handletextpad=.4, borderaxespad=.15)
        card = fig.add_axes([left+.195, .190, .230, .232])
        rows = parameter_card(card, summary, meta, effective)
        if 'best_neutral' in summary:
            fig.text(left, .163, neutral_description(summary['best_neutral']), fontsize=8.2)
        reports[key] = {
            'source_sha256': {str(p): sha256(p) for p in inputs},
            'effective_simulation_parameters': effective, 'displayed_parameter_rows': rows,
            'inference_calibration': meta['calibration'], 'validation_scores': summary['validation'],
            'validation_replicates': len(values), 'bins': len(x),
            'x_coordinate_convention': 'Bin midpoint in zero-based Mb; selected-site annotation is one-based bp',
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
    n = manifest['settings']['validation_replicates']
    fit_bins = manifest['settings'].get('fit_bins', 'train')
    fig.text(.065, .118, r'Simulation: additive genic sweeps (h = 0.5); μ = 1.25 × $10^{-8}$ /bp/generation; deCODE GRCh38 map; 3 Mb simulated, 960 kb shown.', fontsize=8.5)
    fig.text(.065, .091, f'Profiles: means across 20 matched pairs in 20-kb bins; {n} fresh replicates/model. Inference: fixed chromosome Nₑ above; '+r'ρ = $10^{-8}$ /bp/generation.', fontsize=8.5)
    spatial_note = 'all bins fitted; validation uses new seeds' if fit_bins == 'all_bins' else 'spatial blocks withheld; validation uses new seeds'
    fig.text(.065, .064, f'Search: {spatial_note}. Sweep-end time is not allele age. Frequency endpoints are the effective simulated values.', fontsize=8.5)
    fig.text(.065, .037, 'Exploratory fitted configurations, not biological parameter estimates. Ribbons show replicate variation; curves and fitting targets are unchanged.', fontsize=8.5)

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
        'presentation_only': True, 'run_manifest_sha256': sha256(args.run/'manifest.json'),
        'plot_source_sha256': sha256(Path(__file__)), 'yscale': 'log',
        'statistic': 'Arithmetic mean across the frozen 20 pairs; mean across validation replicates',
        'ribbon': '10th–90th percentile across simulation replicates; not pairwise IQR or parameter uncertainty',
        'panels': reports, 'outputs': {str(p.name): sha256(p) for p in args.out.glob('profile_matches.*')},
    }
    (args.out/'figure_metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.out), 'loci': keys, 'array_checks': 'passed'}))


if __name__ == '__main__':
    main()
