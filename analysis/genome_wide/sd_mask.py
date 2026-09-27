"""Unique-base-pair segmental-duplication coverage with explicit coordinates."""
import gzip
from collections import defaultdict


def merge_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end <= start: continue
        if merged and start <= merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else: merged.append((start,end))
    return merged


def load_sd_intervals(path):
    intervals=defaultdict(list)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        for line in stream:
            v=line.rstrip().split('\t')
            if len(v)<4 or not v[1].startswith('chr') or not v[1][3:].isdigit():continue
            intervals[int(v[1][3:])].append((int(v[2]),int(v[3])))
    return {chrom:merge_intervals(rows) for chrom,rows in intervals.items()}


def coverage_fraction(chrom, start, end, intervals):
    """GENCODE 1-based closed gene, UCSC 0-based half-open SD intervals."""
    start0=int(start)-1; end0=int(end)
    if start0<0 or end0<=start0: raise ValueError('Invalid 1-based closed gene interval')
    overlap=0
    for left,right in intervals.get(int(chrom),[]):
        if right<=start0:continue
        if left>=end0:break
        overlap+=max(0,min(end0,right)-max(start0,left))
    return overlap/(end0-start0)


def gene_overlaps_sd(chrom, start, end, intervals, frac_threshold=.5):
    return coverage_fraction(chrom,start,end,intervals)>=frac_threshold
