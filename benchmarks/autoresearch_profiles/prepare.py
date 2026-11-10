"""Freeze audited empirical profiles and independent deCODE generating maps."""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import bin_snps, digest, write_json


def prepare(source, output, key):
    import stdpopsim
    gene, population = key.split('_')
    source = source / key
    meta = json.loads((source / 'COMPLETE.json').read_text())
    if not meta['complete'] or digest(source / 'results.npz') != meta['outputs']['results.npz']:
        raise ValueError('Empirical input failed provenance check')
    raw = np.load(source / 'results.npz', allow_pickle=False)
    calibration = meta['details']['calibration']
    if meta['details']['gamma_metadata']['time_units'] != 'generations':
        raise ValueError('Physical generations required')
    # The audited outputs use VCF one-based POS. Convert once, explicitly.
    center = int(raw['window_start']) - 1 + 500000
    edges = center + np.arange(-480000, 480001, 20000)
    observed = bin_snps(raw['gamma_positions'] - 1, raw['gamma_mean'].mean(axis=1), edges)
    if not np.all(observed > 0):
        raise ValueError('Nonpositive target')
    mid = (edges[:-1] + edges[1:]) / 2
    # Four-bin blocks alternate. Independent simulation seeds provide the
    # main replication test; these spatial blocks are correlated diagnostics.
    train = np.arange(len(mid)) // 4 % 2 == 0
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'target.npz', edges=edges, observed=observed,
        train=train, holdout=~train, all_bins=np.ones(len(mid), dtype=bool),
        focal=np.abs(mid-center) < 100000, pairs=raw['pairs'])
    left, right = center - 1500000, center + 1500000
    gmap = stdpopsim.get_species('HomSap').get_genetic_map('DeCodeSexAveraged_GRCh38')
    chrom = str(raw['chromosome'].item()).removeprefix('chr')
    rate = gmap.get_chromosome_map(chrom).slice(left, right, trim=True)
    if np.any(~np.isfinite(rate.rate)) or np.any(rate.rate < 0):
        raise ValueError('Incomplete recombination map')
    np.savez_compressed(output / 'map.npz', position=rate.position, rate=rate.rate)
    write_json(output / 'manifest.json', dict(key=key, gene=gene, population=population,
        chromosome=chrom, center0=center, left0=left, right0=right,
        gene_interval0=[int(raw['gene_start'])-1, int(raw['gene_end'])],
        n_haplotypes=len(raw['population_hap_indices']), pairs=len(raw['pairs']),
        calibration=calibration, units='generations', assembly='GRCh38',
        input_sha256=digest(source / 'results.npz'), complete_sha256=digest(source / 'COMPLETE.json'),
        target_sha256=digest(output / 'target.npz'), map_sha256=digest(output / 'map.npz'),
        map_id=gmap.id, map_url=gmap.url, map_archive_sha256=gmap.sha256,
        interpretation='Exploratory profile matching; historical 20 pairs, not a population census',
        missing_processes=['BGS', 'multi-population demography', 'phasing error', 'empirical accessibility masks'],
        inference_context='3 Mb simulated context versus full-chromosome empirical context',
        metric='sqrt(0.75*mean(log(sim/obs)^2)+0.25*central200kb_mean(log(sim/obs)^2))'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--keys', nargs='+', default=['GRK2_GIH', 'TREM2_IBS', 'GRK2_YRI', 'TREM2_YRI'])
    a = p.parse_args()
    for key in a.keys:
        prepare(a.source, a.out / key, key)
