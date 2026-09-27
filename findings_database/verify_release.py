#!/usr/bin/env python3
"""Verify release bytes and independently reconstruct its descriptive cascade.

Uses only Python's standard library. Does not rerun raw-VCF preprocessing,
CUDA inference, or reconstruct the duplication union from the original track.
"""
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
GROUPS = ['ACB ASW ESN GWD LWK MSL YRI', 'CEU FIN GBR IBS TSI',
          'CDX CHB CHS JPT KHV', 'BEB GIH ITU PJL STU', 'CLM MXL PEL PUR']
POPS = sorted(' '.join(GROUPS).split())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def value(row, name):
    return float(row[name]) if row[name] else math.nan


def cascade(atlas, reference, threshold):
    counts = [len(atlas)]
    rows = [g for g in atlas if g['is_sd'] == 'False']; counts.append(len(rows))
    rows = [g for g in rows if value(g, 'min_rank') < threshold]; counts.append(len(rows))
    rows = [g for g in rows if any(all(value(g, p+'_rank') < .05 for p in group.split())
                                    for group in GROUPS)]; counts.append(len(rows))
    rows = [g for g in rows if not any(g['chr'] == r['chr']
            and int(g['start']) - 500000 <= int(r['end'])
            and int(g['end']) + 500000 >= int(r['start']) for r in reference)]
    counts.append(len(rows))
    clusters = []
    ordered = sorted(rows, key=lambda g: (int(g['chr']), int(g['start']), int(g['end']), g['gene_id']))
    for _, chromosome in itertools.groupby(ordered, key=lambda g: g['chr']):
        group, end = [], -math.inf
        for g in chromosome:
            if group and int(g['start']) - end > 1000000:
                clusters.append(group); group = []; end = -math.inf
            group.append(g); end = max(end, int(g['end']))
        if group:
            clusters.append(group)
    counts.append(len(clusters))
    representatives = {min(c, key=lambda g: (value(g, 'min_rank'), g['gene_id']))['gene_id'] for c in clusters}
    return counts, representatives


def main():
    latest = json.loads((ROOT / 'LATEST.json').read_text())
    release = ROOT / latest['path']
    manifest = json.loads((release / 'MANIFEST.json').read_text())
    require(latest['release'] == manifest['release'], 'Latest pointer and manifest disagree')
    for name, record in manifest['files'].items():
        path = release / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], f'Checksum: {name}')
        if 'rows' in record:
            require(len(read_csv(path)) == record['rows'], f'Row count: {name}')
    atlas = read_csv(release / 'genome_wide_ranks.csv.gz')
    require(len({g['gene_id'] for g in atlas}) == 19119, 'Gene identities')
    require(sum(g['is_sd'] == 'True' for g in atlas) == 1123, 'SD count')
    for g in atlas:
        require((g['is_sd'] == 'True') == (value(g, 'sd_unique_bp_fraction') >= .5), 'SD threshold')
    for p in POPS:
        eligible = sorted((g for g in atlas if math.isfinite(value(g, p+'_tmrca'))),
                          key=lambda g: value(g, p+'_tmrca'))
        offset = 0
        for _, tied in itertools.groupby(eligible, key=lambda g: value(g, p+'_tmrca')):
            tied = list(tied)
            expected = (offset + 1 + offset + len(tied)) / (2*len(eligible))
            require(all(math.isclose(value(g, p+'_rank'), expected, abs_tol=1e-14) for g in tied), f'Ranks: {p}')
            offset += len(tied)
    reference = read_csv(release / 'reference_loci.csv')
    counts, reps = cascade(atlas, reference, .01)
    require(counts == [19119, 17996, 620, 579, 540, 143], f'Cascade: {counts}')
    require(reps == {g['gene_id'] for g in read_csv(release / 'stage5_loci.csv')}, 'Locus membership')
    for row in read_csv(release / 'threshold_global_sensitivity.csv'):
        actual, _ = cascade(atlas, reference, float(row['cutoff_percent'])/100)
        require(actual == [int(row[f'stage{i}']) for i in range(6)], 'Threshold sensitivity')
    runtime = read_csv(release / 'runtime.csv')
    medians = {s: statistics.median(float(r['seconds']) for r in runtime
               if r['series'] == s and int(r['n_pairs']) == 63190)
               for s in sorted({r['series'] for r in runtime})}
    require(round(medians['cuda'], 2) == 15.91, 'CUDA runtime')
    accuracy = read_csv(release / 'accuracy.csv')
    require(round(statistics.median(value(r, 'r_gamma_smc_cu_median') for r in accuracy), 3) == .833, 'CUDA accuracy')
    require(round(statistics.median(value(r, 'r_gsmc_median') for r in accuracy), 3) == .832, 'CPU accuracy')
    print(json.dumps({'release': latest['release'], 'verified_files': len(manifest['files']),
                      'cascade': counts, 'runtime_medians_seconds': medians,
                      'scope': 'Checksums, row counts, ranks, candidate membership, threshold counts and selected headline summaries.'}, indent=2))


if __name__ == '__main__':
    main()
