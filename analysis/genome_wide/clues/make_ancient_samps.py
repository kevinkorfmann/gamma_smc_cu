#!/usr/bin/env python3
"""
Create CLUES ancientSamps file from AADR genotypes at rs4988235 (LCT).

EIGENSTRAT genotype coding: 0=hom ref (GG), 1=het (GA), 2=hom alt (AA), 9=missing
SNP file says: rs4988235  2  1.555436  136608646  G  A
So ref=G (ancestral), alt=A (derived = LP allele T on opposite strand).

CLUES ancientSamps format: age_generations logP(AA) logP(Aa) logP(aa)
where A=ancestral, a=derived.
  0 = hom ref (GG=AA) -> logP(AA)=0, logP(Aa)=-inf, logP(aa)=-inf
  1 = het (GA=Aa)     -> logP(AA)=-inf, logP(Aa)=0, logP(aa)=-inf
  2 = hom alt (AA=aa) -> logP(AA)=-inf, logP(Aa)=-inf, logP(aa)=0
  9 = missing         -> skip
"""

import csv
import numpy as np
import sys

AADR_DIR = '/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/aadr'
GEN_TIME = 29  # years per generation

# Read sample IDs from .ind file (order matches .geno columns)
sample_ids = []
with open(f'{AADR_DIR}/v62.0_1240k_public.ind') as f:
    for line in f:
        parts = line.strip().split()
        sample_ids.append(parts[0])

# Read ages from .anno file
ages_bp = {}
groups = {}
with open(f'{AADR_DIR}/v62.0_1240k_public.anno') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        gid = row[0]
        try:
            age = float(row[9])
        except (ValueError, IndexError):
            continue
        group = row[13] if len(row) > 13 else ''
        ages_bp[gid] = age
        groups[gid] = group

# Read genotype row
with open(f'{AADR_DIR}/rs4988235_geno_row.txt') as f:
    geno_line = f.read().strip()

print(f"Samples: {len(sample_ids)}, Genotypes: {len(geno_line)}")
assert len(geno_line) == len(sample_ids), f"Mismatch: {len(geno_line)} vs {len(sample_ids)}"

# European-ish keywords for filtering
eur_keywords = ['europe', 'anatolia', 'steppe', 'bell_beaker', 'corded_ware',
                'yamnaya', 'lbk', 'britain', 'iberia', 'scandinavia', 'germany',
                'france', 'hungary', 'czech', 'poland', 'italy', 'greece', 'balkans',
                'ukraine', 'russia', 'nordic', 'celtic', 'viking', 'saxon',
                'beaker', 'funnel', 'linearbandkeramik', 'trypillia',
                'england', 'scotland', 'ireland', 'dutch', 'danish', 'swedish',
                'norwegian', 'finnish', 'estonian', 'latvian', 'lithuanian',
                'romanian', 'bulgarian', 'serbian', 'croatian', 'slovenian',
                'austria', 'switzerland', 'spain', 'portugal', 'sicily', 'sardinia']

# Build ancientSamps file
ancient_diploid = []  # (age_gen, logP_AA, logP_Aa, logP_aa)
ancient_haploid = []  # (age_gen, logP_A, logP_a)

n_anc = 0
n_der = 0
n_het = 0
n_miss = 0

for i, (sid, geno_char) in enumerate(zip(sample_ids, geno_line)):
    if geno_char == '9':
        n_miss += 1
        continue

    age = ages_bp.get(sid)
    if age is None or age < 100:  # skip present-day
        continue

    group = groups.get(sid, '')
    if not any(k in group.lower() for k in eur_keywords):
        continue

    geno = int(geno_char)
    age_gen = age / GEN_TIME  # convert BP to generations

    if geno == 0:  # hom ref = AA (ancestral)
        ancient_diploid.append((age_gen, 0.0, float('-inf'), float('-inf')))
        n_anc += 1
    elif geno == 1:  # het = Aa
        ancient_diploid.append((age_gen, float('-inf'), 0.0, float('-inf')))
        n_het += 1
    elif geno == 2:  # hom alt = aa (derived)
        ancient_diploid.append((age_gen, float('-inf'), float('-inf'), 0.0))
        n_der += 1

# Sort by age
ancient_diploid.sort(key=lambda x: x[0])

print(f"\nAncient European samples with genotype at rs4988235:")
print(f"  Ancestral homozygous (GG): {n_anc}")
print(f"  Heterozygous (GA): {n_het}")
print(f"  Derived homozygous (AA): {n_der}")
print(f"  Missing: {n_miss}")
print(f"  Total usable: {len(ancient_diploid)}")
print(f"  Age range: {ancient_diploid[0][0]*GEN_TIME:.0f} - {ancient_diploid[-1][0]*GEN_TIME:.0f} BP")

# Write ancientSamps file
outfile = f'{AADR_DIR}/LCT_ancientSamps.txt'
with open(outfile, 'w') as f:
    for age_gen, p_AA, p_Aa, p_aa in ancient_diploid:
        f.write(f"{age_gen:.6e} {p_AA:.18e} {p_Aa:.18e} {p_aa:.18e}\n")

print(f"\nWrote {len(ancient_diploid)} samples to {outfile}")

# Also print some frequency estimates by time bin
print("\nDerived allele frequency by era:")
for lo, hi, label in [(0,3000,'Medieval/Recent'), (3000,5000,'Bronze/Iron Age'),
                       (5000,7000,'Late Neolithic'), (7000,9000,'Early Neolithic'),
                       (9000,15000,'Mesolithic'), (15000,50000,'Upper Paleolithic')]:
    samples = [(g, a, b, c) for g, a, b, c in ancient_diploid
               if lo/GEN_TIME <= g < hi/GEN_TIME]
    if samples:
        n_a = sum(1 for _, a, _, _ in samples if a == 0.0)  # hom anc
        n_h = sum(1 for _, _, b, _ in samples if b == 0.0)  # het
        n_d = sum(1 for _, _, _, c in samples if c == 0.0)  # hom der
        freq = (2*n_d + n_h) / (2*len(samples)) if samples else 0
        print(f"  {label:25s}: {len(samples):4d} samples, freq = {freq:.3f}")
