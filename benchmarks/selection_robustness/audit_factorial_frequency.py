"""Audit the completed bank without altering simulations or inference outputs.

Summaries use independent founder clusters. Diversity is weighted by accessible
non-exonic bases; zero-accessibility bins are excluded rather than imputed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from common import digest, write_json


def distribution(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    return dict(n=len(x), mean=float(x.mean()), min=float(x.min()),
                median=float(np.median(x)), p95=float(np.quantile(x, .95)),
                max=float(x.max()))


def measurements(path, focal):
    with np.load(path) as z:
        centers = (z['edges'][1:] + z['edges'][:-1]) / 2
        ans = {}
        for scope, mask in [('genome', np.ones(len(centers), dtype=bool)),
                            ('focal100kb', np.abs(centers-focal) < 50000)]:
            ans[scope+'_tmrca'] = float(np.average(z['tmrca'][mask], weights=np.diff(z['edges'])[mask]))
            valid = mask & (z['accessible_bp'] > 0)
            ans[scope+'_neutral_pi'] = float(np.average(z['neutral_pi'][valid], weights=z['accessible_bp'][valid]))
        return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((a.bank/'manifest.json').read_text())
    complete = json.loads((a.bank/'completion.json').read_text())
    assert len(complete['results']) == 2*manifest['families_per_background']
    assert all(r['status'] == 'completed' for r in complete['results'])
    focal = manifest['region']['focal_position0']
    records, diagnostic_rows, attempts = [], [], []
    sim_count = 0
    seeds = []
    source_sets = defaultdict(set)
    for bg in ['neutral', 'bgs']:
        families = sorted(a.bank.glob(bg+'_family_*'))
        assert len(families) == manifest['families_per_background']
        for fam in families:
            sweeps = sorted(r for r in fam.glob('sweep_*') if r.is_dir())
            assert len(sweeps) == manifest['sweep_attempts_per_family']
            parent_hash = json.loads((fam/'founder/simulation_complete.json').read_text())['final_sha256']
            for run in [fam/'founder', fam/'control', *sweeps]:
                cfg = json.loads((run/'config.json').read_text())
                done = json.loads((run/'simulation_complete.json').read_text())
                processed = json.loads((run/'processed.json').read_text())
                assert cfg['region'] == manifest['region']
                assert cfg['Ne'] == manifest['Ne'] and cfg['mu'] == manifest['mu']
                assert cfg['source_sha256'] == manifest['sources']['factorial_frequency.py']
                assert cfg['scaling_factor'] == 1
                assert cfg['bgs'] == (bg == 'bgs')
                assert done['final_tick'] == cfg['end_tick']
                assert processed['processor_sha256'] == cfg['source_sha256']
                seeds.append(cfg['seed']); sim_count += 1
                if run.name != 'founder':
                    assert cfg['parent_sha256'] == parent_hash
                snaps = {}
                if (run/'snapshots.tsv').exists():
                    for line in (run/'snapshots.tsv').read_text().splitlines():
                        label, tick, freq = line.split('\t')
                        snaps[label] = (int(tick), float(freq))
                if run.name.startswith('sweep_'):
                    tr = np.loadtxt(run/'trajectory.tsv', ndmin=2)
                    # Observation runs both at intro/end callbacks and the
                    # regular late callback. Preserve both original log rows;
                    # intro has pre-introduction 0 then the single-copy value.
                    ticks = np.unique(tr[:, 0])
                    assert np.all(np.diff(ticks) == 1)
                    duplicate_ticks = ticks[np.array([sum(tr[:, 0] == t) for t in ticks]) > 1]
                    assert set(duplicate_ticks).issubset({cfg['start_tick']+1, cfg['end_tick']})
                    for t in duplicate_ticks:
                        values = tr[tr[:, 0] == t, 2]
                        if t == cfg['end_tick']: assert np.all(values == values[0])
                    assert tr[0, 0] == cfg['start_tick']+1
                    assert np.isclose(tr[tr[:, 0] == cfg['start_tick']+1, 2].max(), 1/(2*cfg['Ne']))
                    assert tr[-1, 0] == cfg['end_tick']
                    for name, target in [('f25', .25), ('f50', .5), ('f75', .75), ('f100', 1.)]:
                        crossed = tr[tr[:, 2] >= target, 0]
                        assert bool(len(crossed)) == (name in snaps)
                        if len(crossed):
                            assert crossed[0] == snaps[name][0]
                            assert snaps[name][1] >= target
                    for lag in [250, 1000]:
                        name = f'fixed_plus{lag}'
                        expected = 'f100' in snaps and snaps['f100'][0]+lag <= cfg['end_tick']
                        assert expected == (name in snaps)
                        if expected: assert snaps[name][0]-snaps['f100'][0] == lag
                    attempts.append(dict(background=bg, family=fam.name, run=run.name,
                        outcome='fixed' if tr[-1, 2] == 1 else 'lost' if tr[-1, 2] == 0 else 'segregating',
                        stages={k:dict(tick=v[0], frequency=v[1], age=v[0]-cfg['start_tick']-1) for k,v in snaps.items()}))
                for row in processed['results']:
                    stage = row['stage']
                    assert row['sample_size'] == 50 and row['Ne'] == 10000
                    assert row['tick'] == (done['final_tick'] if stage == 'final' else snaps[stage][0])
                    rec = dict(background=bg, family=fam.name, run=run.name, stage=stage,
                        uncoalesced_span_fraction=row['uncoalesced_span_fraction'],
                        **measurements(run/f'{stage}.npz', focal))
                    records.append(rec)
                    metric_path = run/'gamma'/stage/'metrics.json'
                    m = json.loads(metric_path.read_text())
                    assert m['tree_sha256'] == row['variants_sha256']
                    assert m['source_sha256'] == manifest['sources']['infer_factorial_gamma.py']
                    assert m['pairs'] == 1225 and m['genotype_shape'][0] == 50
                    for mode in ['auto', 'fixed']: assert m[mode]['core_coverage'] == 1
                    for name, sha in m['software_files_sha256'].items(): source_sets[name].add(sha)
                    diagnostic_rows.append(dict(background=bg, family=fam.name, run=run.name, stage=stage,
                        total_snps=m['total_snps'], focal_snps=m['focal100kb_snps'],
                        auto=m['auto'], fixed=m['fixed'], context=m.get('context_sensitivity', {})))
    assert sim_count == 2*manifest['families_per_background']*(2+manifest['sweep_attempts_per_family'])
    assert len(seeds) == len(set(seeds))
    assert all(len(s) == 1 for s in source_sets.values())
    rng = np.random.default_rng(260926); B = 4000
    boot = {}; point = {}; ns = {}
    metrics = ['genome_tmrca', 'focal100kb_tmrca', 'genome_neutral_pi', 'focal100kb_neutral_pi']
    root_summary = []
    for bg in ['neutral', 'bgs']:
        families = sorted({r['family'] for r in records if r['background'] == bg})
        weights = rng.multinomial(len(families), np.full(len(families), 1/len(families)), size=B)
        groups = defaultdict(list)
        for r in records:
            if r['background'] != bg: continue
            kind = r['run'] if r['run'] in ['founder', 'control'] else 'sweep'
            groups[(kind, r['stage'])].append(r)
        for (kind, stage), rows in groups.items():
            key = (bg, kind, stage)
            numbers = np.array([sum(r['family'] == f for r in rows) for f in families])
            denom = weights @ numbers
            ns[key] = dict(snapshots=len(rows), contributing_families=int((numbers > 0).sum()))
            root_summary.append(dict(background=bg, kind=kind, stage=stage,
                uncoalesced_span_fraction=distribution([r['uncoalesced_span_fraction'] for r in rows])))
            for metric in metrics:
                sums = np.array([sum(r[metric] for r in rows if r['family'] == f) for f in families])
                boot[key+(metric,)] = np.divide(weights @ sums, denom,
                    out=np.full(B, np.nan), where=denom > 0)
                point[key+(metric,)] = float(sums.sum()/numbers.sum())
    contrasts = []
    def compare(name, numerator, denominator):
        for metric in metrics:
            nk, dk = numerator+(metric,), denominator+(metric,)
            ratio = boot[nk]/boot[dk]
            contrasts.append(dict(contrast=name, metric=metric,
                ratio=point[nk]/point[dk], ci95=np.nanquantile(ratio, [.025, .975]).tolist(),
                numerator_mean=point[nk], denominator_mean=point[dk],
                numerator_n=ns[numerator], denominator_n=ns[denominator]))
    compare('BGS / neutral control', ('bgs','control','final'), ('neutral','control','final'))
    compare('BGS burn 10Ne / 5Ne', ('bgs','founder','final'), ('bgs','founder','burn5Ne'))
    for bg in ['neutral', 'bgs']:
        for stage in ['control_t100', 'control_t300', 'control_t1000', 'final']:
            compare(bg+' control '+stage+' / founder', (bg,'control',stage), (bg,'founder','final'))
        for stage in ['f25','f50','f75','f100','fixed_plus250','fixed_plus1000']:
            if (bg,'sweep',stage) in ns:
                compare(bg+' '+stage+' / control', (bg,'sweep',stage), (bg,'control','final'))
    inference_summary = []
    dg = defaultdict(list)
    for r in diagnostic_rows:
        kind = r['run'] if r['run'] in ['founder','control'] else 'sweep'
        dg[(r['background'],kind,r['stage'])].append(r)
    for (bg,kind,stage), rows in dg.items():
        inference_summary.append(dict(background=bg,kind=kind,stage=stage,
            total_snps=distribution([r['total_snps'] for r in rows]),
            focal_snps=distribution([r['focal_snps'] for r in rows]),
            errors={mode:{metric:distribution([r[mode][metric] for r in rows])
                for metric in ['mean_log_error','rmse_log']} for mode in ['auto','fixed']},
            context={crop:{metric:distribution([r['context'][crop][metric] for r in rows if crop in r['context']])
                for metric in ['median_relative_change','p95_relative_change']}
                for crop in ['auto_3016000','auto_6000000','fixed_3016000','fixed_6000000']}))
    report = dict(source_sha256=digest(__file__), manifest_sha256=digest(a.bank/'manifest.json'),
        checks_passed=True, simulations=sim_count, inference_snapshots=len(diagnostic_rows),
        introductions=len(attempts), unique_seeds=len(set(seeds)),
        software_hashes={k:list(v)[0] for k,v in source_sets.items()},
        checks=['Immutable region, map, Ne, mu, simulator and inference sources',
                'Correct parent checkpoints, final ticks, and unique seeds',
                'All 720 single-copy introductions and natural first passages',
                'Post-fixation timing and all 50-sample/1225-pair inference snapshots',
                'Full supported projection coverage in scoring interval'],
        contrasts=contrasts, root_diagnostics=root_summary, inference_diagnostics=inference_summary,
        bootstrap_replicates=B, bootstrap_seed=260926,
        uncertainty='Founder clusters; paired resampling within background, independent across backgrounds')
    write_json(a.out/'audit_summary.json', report)
    write_json(a.out/'biological_measurements.json', records)
    write_json(a.out/'all_inference_metrics.json', diagnostic_rows)
    write_json(a.out/'all_introductions.json', attempts)
    print(json.dumps({k:report[k] for k in ['checks_passed','simulations','inference_snapshots','introductions']}))


if __name__ == '__main__': main()
