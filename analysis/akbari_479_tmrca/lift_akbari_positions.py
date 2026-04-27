#!/usr/bin/env python3
"""Lift Akbari 2026 lead-variant positions from GRCh37 to GRCh38.

Akbari 2026 Harvard Dataverse (doi:10.7910/DVN/7RVV9N) publishes positions
on GRCh37. Our 1000G 30x cache is on GRCh38 (Byrska-Bishop 2022). This script
produces akbari_lead_variants_grch38.tsv with the POS column lifted to GRCh38
and the original GRCh37 position preserved in POS_GRCh37.
"""
import csv, sys, os
from pyliftover import LiftOver

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'akbari_lead_variants.tsv')
DST  = os.path.join(HERE, 'akbari_lead_variants_grch38.tsv')
CHAIN = '/vast/projects/smathi/cohort/kkor/tmrca.cu/references/hg19ToHg38.over.chain.gz'

lo = LiftOver(CHAIN)

with open(SRC) as fin, open(DST, 'w', newline='') as fout:
    reader = csv.reader(fin, delimiter='\t')
    writer = csv.writer(fout, delimiter='\t')
    header = next(reader)
    assert header[0] == 'CHROM' and header[1] == 'POS', f'unexpected header {header}'
    new_header = ['CHROM', 'POS'] + header[2:] + ['POS_GRCh37']
    writer.writerow(new_header)

    n_total = n_lifted = n_failed = n_cross_chr = 0
    failed_rows = []
    for row in reader:
        n_total += 1
        chrom_src = row[0]
        pos_src = int(row[1])
        res = lo.convert_coordinate(f'chr{chrom_src}', pos_src - 1)  # pyliftover is 0-based
        if not res:
            n_failed += 1
            failed_rows.append((chrom_src, pos_src, row[2]))
            continue
        new_chrom, new_pos0, _strand, _score = res[0]
        new_chrom_short = new_chrom[3:] if new_chrom.startswith('chr') else new_chrom
        if new_chrom_short != chrom_src:
            n_cross_chr += 1
            failed_rows.append((chrom_src, pos_src, row[2]))
            continue
        n_lifted += 1
        new_pos = new_pos0 + 1
        writer.writerow([new_chrom_short, new_pos] + row[2:] + [pos_src])

print(f'Total input: {n_total}', file=sys.stderr)
print(f'Lifted OK (same chrom): {n_lifted}', file=sys.stderr)
print(f'Failed (no mapping): {n_failed}', file=sys.stderr)
print(f'Cross-chromosome remap (dropped): {n_cross_chr}', file=sys.stderr)
for (c, p, r) in failed_rows[:20]:
    print(f'  DROPPED: chr{c}:{p} {r}', file=sys.stderr)
