"""Plot all-replicate sweep signatures from independently verified tree summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

REGIMES = ['incomplete', 'near_fixed', 'completed', 'older_completed']
TITLES = ['Incomplete', 'Near fixation', 'Completed', 'Older completed']
DETAILS = ['Selected-allele frequency: 25–75%', 'Selected-allele frequency: 80–99%',
           'Median fixation: 688 generations ago', 'Median fixation: 2,688 generations ago']
COLORS = {'true_tmrca': '#252525', 'inferred_tmrca': '#2166AC',
          'ihs_max': '#C65C2E', 'xp_max': '#208878'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(a):
    source = a.profiles / 'profiles_summary.csv'
    verification = json.loads((a.profiles / 'verification.json').read_text())
    assert verification['status'] == 'passed'
    assert sha(source) == verification['outputs'][source.name]
    data = pd.read_csv(source)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': 'black', 'axes.labelcolor': 'black'})
    fig = plt.figure(figsize=(7.2, 6.3))
    grid = fig.add_gridspec(2, 2, left=.14, right=.983, bottom=.105,
                            top=.855, wspace=.25, hspace=.53)
    neutral = data[data.regime == 'neutral'].sort_values('offset_bp')
    counts = {}
    for i, regime in enumerate(REGIMES):
        block = data[data.regime == regime].sort_values('offset_bp')
        assert len(block) == 101 and (block.n == 100).all()
        inner = grid[i // 2, i % 2].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=.08)
        top = fig.add_subplot(inner[0]); bottom = fig.add_subplot(inner[1], sharex=top)
        x = block.offset_bp.to_numpy() / 1000
        for ax, metrics in [(top, ['true_tmrca', 'inferred_tmrca']),
                            (bottom, ['ihs_max', 'xp_max'])]:
            ax.axvline(0, color='#929292', lw=.85, ls=':', zorder=0)
            for metric in metrics:
                ax.fill_between(x, block[metric + '_q25'].to_numpy(),
                    block[metric + '_q75'].to_numpy(), color=COLORS[metric], alpha=.12, lw=0)
                ax.plot(x, block[metric + '_median'], color=COLORS[metric], lw=1.65,
                        ls='--' if metric in ['true_tmrca', 'ihs_max'] else '-', zorder=3)
            ax.set_xlim(-250, 250)
            ax.set_xticks([-200, -100, 0, 100, 200])
            ax.tick_params(length=3, pad=2)
            ax.grid(axis='y', color='#e8e8e8', lw=.45)
        top.plot(neutral.offset_bp / 1000, neutral.true_tmrca_median,
                 color='#888888', ls=':', lw=1.2, zorder=1)
        top.set_yscale('log'); top.set_ylim(650, 18000)
        top.set_yticks([1000, 3000, 10000], ['1,000', '3,000', '10,000'])
        top.minorticks_off(); top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        bottom.set_ylim(0, 7); bottom.set_yticks([0, 2, 4, 6])
        if i % 2 == 0:
            top.set_ylabel('TMRCA\n(generations)', labelpad=4)
            bottom.set_ylabel('Haplotype\nscore', labelpad=5)
        top.text(-.01, 1.30, 'abcd'[i], transform=top.transAxes, weight='bold', fontsize=11)
        top.text(.07, 1.30, TITLES[i], transform=top.transAxes, fontsize=10, weight='bold')
        top.text(.07, 1.10, DETAILS[i], transform=top.transAxes, fontsize=7.8)
        centre = block[block.offset_bp == 0].iloc[0]
        counts[regime] = int(centre.ihs_max_n)
        bottom.text(.98, .06, f'iHS available: {int(centre.ihs_max_n)}/100',
                    transform=bottom.transAxes, ha='right', fontsize=7.5,
                    bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 1})
        if i >= 2:
            bottom.set_xlabel('Distance from selected site (kb)', labelpad=4)
    first = [Line2D([], [], color=COLORS['true_tmrca'], ls='--', lw=1.7, label='True TMRCA'),
             Line2D([], [], color=COLORS['inferred_tmrca'], lw=1.7, label='Gamma-SMC (CPU)'),
             Line2D([], [], color='#888888', ls=':', lw=1.3, label='Neutral true TMRCA')]
    second = [Line2D([], [], color=COLORS['ihs_max'], ls='--', lw=1.7, label='Maximum |iHS|'),
              Line2D([], [], color=COLORS['xp_max'], lw=1.7, label='Maximum XP-EHH')]
    fig.legend(handles=first, loc='upper center', bbox_to_anchor=(.54, 1.006),
               ncol=3, frameon=False, fontsize=8.4, columnspacing=1.3)
    fig.legend(handles=second, loc='upper center', bbox_to_anchor=(.54, .967),
               ncol=2, frameon=False, fontsize=8.4, columnspacing=2)
    fig.text(.14, .022, 'Lines: medians. Shading: middle 50% of replicates. Dotted vertical line: selected site.', fontsize=7.6)
    a.out.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(a.out / f'fig_sweep_signatures.{extension}', dpi=400, facecolor='white')
    plt.close(fig)
    receipt = {'source_sha256': sha(source), 'plot_source_sha256': sha(__file__),
        'histories': REGIMES, 'replicates_per_history': 100, 'neutral_replicates': 200,
        'centre_ihs_computable': counts, 'width_bp': 20000, 'step_bp': 5000,
        'aggregation': 'Median and IQR among finite scores, with missing values retained as missing',
        'outputs': {f.name: sha(f) for f in a.out.glob('fig_sweep_signatures.*')
                    if f.suffix in ['.pdf', '.png', '.svg']}}
    (a.out / 'fig_sweep_signatures.json').write_text(json.dumps(receipt, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
