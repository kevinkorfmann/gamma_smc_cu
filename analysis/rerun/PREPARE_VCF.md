# Strict VCF input preparation

Rebuild all chromosomes from the same original phased cohort VCF release. Do
not mix historical caches with the new filtered input stream. This command is
CPU-only; on Betty run it in a Slurm allocation.

```sh
python analysis/rerun/prepare_vcf.py \
  --vcf /original-vcfs/chr22.vcf.gz \
  --output /new-inputs/cache/parsed/chr22 --chr 22 \
  --expected-samples 3202 --threads 18 --chunk-sites 8192
```

Requirements are numpy and cyvcf2. The sample count defaults to 3,202 and can
be explicitly overridden for a different cohort. Samples retain their VCF
order; each contributes its first and second GT alleles as consecutive
haplotype rows. Every sample ID must be unique. Input positions must be
positive and sorted on the requested chromosome.

Every record at a position that occurs more than once is removed **before**
other filtering. Thus a valid SNP sharing its position with an indel is also
removed. Remaining records must be PASS or dot, have distinct single-base
A/C/G/T REF and ALT alleles, and have complete diploid 0/1 genotypes in every
sample. Every heterozygote must be phased; unphased homozygotes are accepted
because swapping their alleles has no effect. Cohort-wide monomorphic SNP
records are retained. This explicit filter changes the data relative to the
historical multiallelic/duplicated-position cache.

The cache contains `G.npy` (`uint8`, haplotypes × sites), `positions.npy`
(`int64`) and `sample_ids.npy` (Unicode, no pickle needed). `READY.json`
records original VCF SHA256, all member hashes, filter counts, versions,
source-code hash, dimensions, sample order convention and coordinate system.
Positions are exact **one-based VCF POS**, with numeric `positions_base: 1`,
matching the empirical pipeline's one-based inclusive gene intervals. They
are never cast through float32 or silently shifted. The optional VCF contig
length is retained in metadata. Genome-wide inference currently keeps its
explicitly documented historical theta denominator of last retained POS + 1
unless `--sequence-length` is supplied; that convention is separate from the
position coordinate system.

Two passes apply identical filters. The first counts retained sites; the
second writes chunked NPY memory maps. With 6,404 haplotypes and 8,192-site
chunks, the genotype write buffer is about 52 MB. Mapped output pages and
filesystem cache are additional OS-managed memory. No full chromosome
genotype array is loaded into process RAM. The destination is created
atomically only after both passes agree and outputs are hashed. Existing
outputs are never replaced. `--resume` accepts only the identical source,
filter/source-code identity and verified output hashes; inspect and move
failed or mismatched caches aside explicitly before rerunning.

The shared loader `analysis.rerun.cache.load_chromosome(cache_root, chromosome)`
already reads this layout as genuine memory maps. Do not copy a READY marker
without all three matching members.

```sh
python -m pytest -q analysis/rerun/tests/test_prepare_vcf.py
```

Tests parse actual miniature VCF files and check duplicate/indel interactions,
phasing and missingness, exact large positions, sample/haplotype ordering,
monomorphic and unphased-homozygous records, cohort/chromosome checks,
non-monotonic positions, and checksum-verified resumption.
