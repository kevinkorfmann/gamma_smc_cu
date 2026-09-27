from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'prepare_vcf.py'
spec = importlib.util.spec_from_file_location('prepare_vcf', MODULE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def write_vcf(path, rows):
    path.write_text('##fileformat=VCFv4.2\n'
                    '##contig=<ID=22,length=300000000>\n'
                    '##FILTER=<ID=q10,Description="Low quality">\n'
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n'
                    + ''.join('22\t' + '\t'.join(map(str, row)) + '\n' for row in rows))


def row(pos, ref='A', alt='C', filt='PASS', a='0|1', b='1|0'):
    return (pos, '.', ref, alt, '.', filt, '.', 'GT', a, b)


def test_vcf_cache_filters_coordinates_layout_and_resume(tmp_path):
    pytest.importorskip('cyvcf2')
    source = tmp_path / 'input.vcf'
    rows = [row(10), row(20), row(20, alt='CA'),  # remove BOTH duplicate rows
            row(30, filt='q10'), row(40, ref='N'), row(50, alt='C,G'),
            row(60, alt='<DEL>'), row(70, a='./.'), row(80, a='0/1'),
            row(90, a='0/0', b='1/1'),  # unphased homozygotes are unambiguous
            row(100, a='0'), row(110, a='0|2'),
            row(120, filt='.', a='0|0', b='0|0'),  # monomorphic input retained
            row(16777217, a='1|0', b='0|1')]
    write_vcf(source, rows)
    output = tmp_path / 'cache' / 'chr22'
    record = p.prepare(source, output, 22, expected_samples=2, chunk_sites=2)
    G = np.load(output / 'G.npy', mmap_mode='r')
    pos = np.load(output / 'positions.npy')
    assert pos.dtype == np.int64 and pos.tolist() == [10, 90, 120, 16777217]
    assert G.dtype == np.uint8 and G.shape == (4, 4)
    assert G.tolist() == [[0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]]
    assert np.load(output / 'sample_ids.npy', allow_pickle=False).tolist() == ['A', 'B']
    counts = record['filter_counts']
    assert counts['input_records'] == 14
    assert counts['duplicate_position_records'] == 2
    assert counts['duplicate_position_groups'] == 1
    assert counts['retained_sites'] == 4
    assert counts['retained_unphased_homozygous_genotypes'] == 2
    assert counts['rejected_filter'] == 1
    assert counts['rejected_not_biallelic_acgt_snp'] == 3
    assert counts['rejected_unphased_heterozygote'] == 1
    assert record['vcf_contig_length_bp'] == 300000000
    assert record['source_sha256'] == p.sha256(source)
    assert p.prepare(source, output, 22, expected_samples=2, resume=True) == record
    with pytest.raises(FileExistsError):
        p.prepare(source, output, 22, expected_samples=2)
    writable = np.load(output / 'G.npy', mmap_mode='r+')
    writable[0, 0] = 1
    writable.flush()
    with pytest.raises(FileExistsError):
        p.prepare(source, output, 22, expected_samples=2, resume=True)


@pytest.mark.parametrize('rows,error', [([row(20), row(10)], 'nondecreasing'),
                                         ([row(20), row(20)], 'No sites retained')])
def test_reject_unsorted_or_all_duplicates(tmp_path, rows, error):
    pytest.importorskip('cyvcf2')
    source = tmp_path / 'input.vcf'
    write_vcf(source, rows)
    output = tmp_path / 'chr22'
    with pytest.raises(ValueError, match=error):
        p.prepare(source, output, 'chr22', expected_samples=2)
    assert not output.exists()


def test_reject_wrong_cohort_or_chromosome(tmp_path):
    pytest.importorskip('cyvcf2')
    source = tmp_path / 'input.vcf'
    write_vcf(source, [row(20)])
    with pytest.raises(ValueError, match='expected 3202'):
        p.prepare(source, tmp_path / 'cohort', 22)
    with pytest.raises(ValueError, match='Unexpected chromosome'):
        p.prepare(source, tmp_path / 'chromosome', 21, expected_samples=2)
