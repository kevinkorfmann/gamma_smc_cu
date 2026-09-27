#!/usr/bin/env python3
"""Plot measured, hash-matched end-to-end runtimes; never extrapolate endpoints."""
import argparse
import json
import math
from pathlib import Path

SERIES = {
    'cuda': ('gamma_smc_cu · 1 V100S', '#009E73', 'o', '-'),
    'gamma_cpu': ('gamma_smc · 1 CPU process', '#0072B2', '^', '--'),
    'asmc8': ('ASMC · 8 CPU workers', '#CC79A7', 'D', '-.'),
    'asmc1': ('ASMC · 1 CPU worker', '#D55E00', 's', ':'),
}


def validate_records(records):
    if not records or {r['series'] for r in records} != set(SERIES):
        raise ValueError('Require CUDA, CPU Gamma-SMC, and ASMC with 1 and 8 workers.')
    panels = {(r['panel_manifest_sha256'], r['n_sites'], r['n_haplotypes'], r['host']) for r in records}
    if len(panels) != 1:
        raise ValueError('All methods must use the identical full panel and host.')
    groups = {key: {} for key in SERIES}
    identities = {}
    for row in records:
        count = row['n_pairs']
        if count < 1 or count in groups[row['series']]:
            raise ValueError('Pair counts must be positive and unique within each series.')
        if row['status'] != 'ok' or row['is_extrapolated']:
            raise ValueError('Only completed measured results may be plotted.')
        if row['metric'] != 'native_total' or not row['includes_output']:
            raise ValueError('Require end-to-end native wall time including output.')
        if len(row['seconds']) != 3 or any(not math.isfinite(t) or t <= 0 for t in row['seconds']):
            raise ValueError('Require three positive finite measured repeats, excluding warmup.')
        identity = (row['pair_set'], row['pairs_sha256'])
        if count in identities and identities[count] != identity:
            raise ValueError('Pair identity/order differs between methods.')
        identities[count] = identity
        groups[row['series']][count] = row
    expected = set(groups['cuda'])
    if any(set(group) != expected for group in groups.values()):
        raise ValueError('Each series must cover exactly the same measured pair counts.')
    n_haps = records[0]['n_haplotypes']
    if max(expected) != n_haps*(n_haps-1)//2 or identities[max(expected)][0] != 'all':
        raise ValueError('The all-pairs endpoint must actually cover every distinct pair.')
    return groups


def plot(records, prefix, *, return_figure=False):
    groups = validate_records(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, (scaling, endpoint) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                           gridspec_kw={'width_ratios': [1.4, 1]})
    counts = sorted(groups['cuda'])
    for key, (label, color, marker, style) in SERIES.items():
        values = np.array([groups[key][n]['seconds'] for n in counts])
        medians = np.median(values, axis=1)
        scaling.errorbar(counts, medians,
            yerr=[medians-values.min(axis=1), values.max(axis=1)-medians],
            color=color, marker=marker, linestyle=style, markersize=4,
            linewidth=1.35, elinewidth=.8, capsize=2, label=label)
    scaling.set(xscale='log', yscale='log', xlabel='Distinct haplotype pairs',
                ylabel='Native total elapsed time (s)')
    scaling.set_xticks([counts[0], 1000, 10000, counts[-1]])
    scaling.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    scaling.tick_params(axis='x', rotation=25)
    scaling.grid(axis='y', which='major', color='0.92', linewidth=.6)
    scaling.set_axisbelow(True)

    baseline = float(np.median(groups['cuda'][counts[-1]]['seconds']))
    for y, (key, (label, color, marker, style)) in enumerate(SERIES.items()):
        values = np.array(groups[key][counts[-1]]['seconds'])
        median = np.median(values)
        endpoint.plot([values.min(), values.max()], [y, y], color=color, linewidth=1.5)
        endpoint.scatter(values, y+np.array([-.09, 0, .09]), color=color, s=13,
                         marker=marker, alpha=.6, zorder=3)
        endpoint.scatter([median], [y], color=color, s=31, marker=marker, zorder=4)
        text = (f'{median:.2f} s' if median < 60 else
                f'{median/60:.1f} min' if median < 3600 else f'{median/3600:.2f} h')
        if key != 'cuda':
            text += f'  ({median/baseline:.0f}×)'
        endpoint.annotate(text, (median, y), xytext=(0, -17), textcoords='offset points',
                          ha='center', fontsize=7, color='black')
    endpoint.set(xscale='log', xlabel='Native total elapsed time (s)',
                 ylim=(3.6, -.45))
    endpoint.set_yticks(range(len(SERIES)), ['CUDA', 'gamma_smc', 'ASMC (8)', 'ASMC (1)'])
    endpoint.set_xlim(baseline*.45, max(groups['asmc1'][counts[-1]]['seconds'])*4)
    endpoint.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    endpoint.grid(axis='x', which='major', color='0.92', linewidth=.6)
    endpoint.set_axisbelow(True)

    for ax, label in zip((scaling, endpoint), ('A', 'B')):
        ax.text(-.16, 1.08, label, transform=ax.transAxes, weight='bold', fontsize=11)
    handles, labels = scaling.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=2, frameon=False, fontsize=7.5, handlelength=2.6, columnspacing=2)
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.28, wspace=.58)
    if return_figure:
        return fig
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(prefix.with_suffix('.'+suffix), dpi=400, facecolor='white')
        if suffix == 'svg':
            svg = prefix.with_suffix('.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.records.read_text())['records'], args.output_prefix)


if __name__ == '__main__':
    main()
