#!/usr/bin/env python3
"""Create the 572 approved production genome/lead-bank tasks; submit nothing."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

POPS = sorted('ACB ASW ESN GWD LWK MSL YRI CEU FIN GBR IBS TSI CDX CHB CHS JPT KHV BEB GIH ITU PJL STU CLM MXL PEL PUR'.split())
CHROMOSOME_PRIORITY = [22, 11, 6, 12, 20, 2]
POPULATION_PRIORITY = ['ASW', 'YRI', 'CEU']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', required=True, help='The production frozen SOURCE_MANIFEST.json')
    p.add_argument('--source', default='source-v2', help='Frozen production source directory relative to run root')
    p.add_argument('--output', required=True, help='New worker task manifest')
    p.add_argument('--lead-variants', default='{inputs}/akbari_lead_variants_grch38.tsv')
    p.add_argument('--production-approval', default='{root}/validation/production-approved.json')
    p.add_argument('--core-block-sites', type=int, default=65536)
    p.add_argument('--flank-sites', type=int, default=8192)
    p.add_argument('--pair-chunk', type=int, default=512)
    p.add_argument('--lead-half-bp', type=int, default=25000)
    p.add_argument('--mu', type=float, default=1.25e-8)
    p.add_argument('--rho', type=float, default=1e-8)
    return p


def build(a):
    source = Path(a.source)
    if source.is_absolute() or '..' in source.parts or str(source) == '.':
        raise ValueError('Source must be a safe relative source directory')
    if min(a.core_block_sites, a.pair_chunk, a.lead_half_bp) < 1 or a.flank_sites < 0 or a.mu <= 0 or a.rho < 0:
        raise ValueError('Invalid production inference settings')
    frozen = json.loads(Path(a.source_manifest).read_text())
    required_source = 'analysis/genome_wide/infer_chromosome.py'
    if required_source not in frozen.get('files', {}):
        raise ValueError('Production source manifest lacks the chromosome inference script')
    settings = dict(mu=a.mu, rho=a.rho, core_block_sites=a.core_block_sites,
                    flank_sites=a.flank_sites, pair_chunk=a.pair_chunk, lead_half_bp=a.lead_half_bp,
                    time_units='generations', positions_base=1, positions_dtype='int64',
                    inference_context='full chromosome for both genes and lead windows',
                    sequence_length_convention='last retained VCF POS + 1',
                    pair_selection='all unordered focal-population haplotype pairs')
    chromosomes = CHROMOSOME_PRIORITY + [c for c in range(1, 23) if c not in CHROMOSOME_PRIORITY]
    populations = POPULATION_PRIORITY + [p for p in POPS if p not in POPULATION_PRIORITY]
    tasks = []
    for chrom in chromosomes:
        for pop in populations:
            command = ['{python}', '{source}/analysis/genome_wide/infer_chromosome.py',
                       '--chr', str(chrom), '--populations', pop,
                       '--cache-dir', '{inputs}/cache', '--samples', '{inputs}/samples.txt',
                       '--output-dir', '{output}/results',
                       '--core-block-sites', str(a.core_block_sites), '--flank-sites', str(a.flank_sites),
                       '--pair-chunk', str(a.pair_chunk), '--mu', str(a.mu), '--rho', str(a.rho),
                       '--lead-variants', a.lead_variants, '--lead-half-bp', str(a.lead_half_bp)]
            tasks.append(dict(id=f'genome_chr{chrom}_{pop}', kind='gpu', stage='genome', command=command,
                requires=[f'{{inputs}}/cache/parsed/chr{chrom}/READY.json',
                          f'{{inputs}}/cache/genes/chr{chrom}_genes.tsv', '{inputs}/samples.txt',
                          a.lead_variants, a.production_approval],
                expected=[f'results/chr{chrom}/{pop}{ext}' for ext in ('.npz', '.csv', '.metadata.json')],
                resources=dict(cpus=4, memory_gb=96, walltime_hours=48, dedicated=False)))
    return dict(schema=1, source=a.source, source_manifest_sha256=sha256(a.source_manifest),
                scientific_settings=settings,
                validation_gate=a.production_approval,
                collection=dict(task_id_pattern='genome_chrN_POP', source_directory=a.source,
                                expected_gene_results=572, cross_host_validation_required=True),
                summary=dict(tasks=len(tasks), chromosomes=chromosomes, populations=populations), tasks=tasks)


def main():
    args = parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace production manifest {output}')
    manifest = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
