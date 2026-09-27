# Fresh Relate and CLUES2 reruns

`relate_clues.py` runs complete chromosome genealogy inference and population
size estimation before sampling focal trees and running CLUES2. Use a new output
root on each host. The source repository supplies retained raw inputs and tool
installations; historical trees and CLUES outputs are never read.

```bash
python analysis/rerun/relate_clues.py manifest \
  --source-repo /path/to/retained/gamma_smc_cu \
  --output /path/to/new/relate_clues \
  --threads 16 --memory-gb 128 --seed 42
```

Paths can be overridden with `--relate-root`, `--clues-root`, `--reference-root`,
`--vcf-dir`, and `--samples`. Manifest creation checks files but does not perform
genomic computation. The manifest prints task names. Run the stages in this
order; tasks within a stage can run separately:

```bash
python analysis/rerun/relate_clues.py run --manifest /path/to/new/relate_clues/manifest.json \
  --stage prepare --task chr11_EURSAS
python analysis/rerun/relate_clues.py run --manifest /path/to/new/relate_clues/manifest.json \
  --stage relate --task chr11_EURSAS
python analysis/rerun/relate_clues.py run --manifest /path/to/new/relate_clues/manifest.json \
  --stage popsize --task chr11_EURSAS_GIH
python analysis/rerun/relate_clues.py run --manifest /path/to/new/relate_clues/manifest.json \
  --stage locus --task GRK2
```

On Betty, submit each `run` command through Slurm; on Sesame run it directly.
`prepare` requires input conversion and chromosome-sized memory. Historical
Relate requests used 192 GB RAM and 48 hours for chromosome inference. CLUES
memory scales strongly with frequency grid size, time cutoff, haplotypes and
tree count; use dedicated CPU jobs with sufficient RAM. Defaults are 200
branch samples, 200 frequency grid points, 2,000 generations, ten population
size iterations, 100 trajectory-integration points, mutation rate 1.25e-8 and
28 years/generation. These are recorded, configurable values, not convergence
guarantees. `selection_diagnostics.json` flags estimates within 5% of `sMax`;
the default bound is 0.1, so flagged results require a separate wider-bound
sensitivity run before interpretation.

Transfer these retained inputs for chromosomes 2, 6, 11, 12 and 20:

- `analysis/genome_wide/data/chrN.vcf.gz`, its `.tbi` or `.csi`, and `samples.txt`.
- `analysis/relate_clues/tools/relate_v1.2.4/` including binaries and scripts.
- `analysis/relate_clues/tools/CLUES2/`. The inspected Betty checkout was
  `b20dc5df6b8e6c93a1cfaf2ea6d0f09d04f4c52b`; each stage hashes executed core files.
- Under `input_files/Relate_input_files/GRCh38/`, the chromosome's
  `human_ancestor_GRCh38/homo_sapiens_ancestor_N.fa.gz`,
  `20160622_genome_mask_GRCh38/StrictMask/20160622.chrN.mask.fasta.gz`, and
  `recomb_map/genetic_map_chrN.txt`.

Install the CLUES Python dependencies (`numpy`, `scipy`, `pandas`, `numba`,
`biopython`) and `pysam` in the interpreter used to launch the driver. The
driver uses that same interpreter for CLUES. Population-size plots are disabled
so R is not required. The original Relate directory structure must be retained
because its scripts locate their binaries relative to themselves.

The manifest includes seven case-study loci, LCT positive control and five
prespecified neutral controls. Seven TREM2 anchors are separate tasks and are
all reported. IFIH1 uses IBS, matching the focal population in the manuscript;
older CLUES scripts used CEU. CDX receives its own population-size fit rather
than borrowing the CHS coalescence history. Consequently these are corrected
reruns and need not reproduce historical point estimates.

The pipeline determines the derived base from the ancestral FASTA and original
VCF REF/ALT. Ambiguous bases, unmatched alleles, missing/unphased genotypes,
monomorphic targets, or inconsistent haplotype order fail the locus. Approximate
historical anchors choose the nearest valid SNP within the declared 5 kb limit
before selection inference; exact GRK2/LCT/TREM2 anchors cannot silently move.
Every eligible/rejected candidate and the selected position are recorded.
Only the focal local tree from each MCMC sample enters CLUES, even though
branch lengths are sampled over a flanking region. Polarization is checked
again against the sampled `.sites` nucleotide strings.

CLUES2's converter can change individual derived states to enforce a single
mutation on a tree. Both the original and converted vectors, changed leaf IDs,
and their frequencies are recorded. The default
`--max-converter-flip-fraction 0` rejects any such changes. An explicitly
declared nonzero tolerance preserves the measured present-day derived frequency
for CLUES and documents its difference from the converter's vector. Do not
interpret a failed locus as evidence against selection or retry targets based
on their selection scores.

CLUES2 has no random-seed CLI option for its trajectory integration. A generated
launcher seeds Python and NumPy before invoking the unmodified inference
script. The posterior grid is retained, and `result_trajectory95.tsv` gives
posterior means, medians and equal-tail 95% intervals at each generation, with
a separate years column. These intervals are distinct from CLUES's `_CI.txt`,
which reports intervals for selection coefficients.

Each attempt records command arguments, logs, source/tool hashes, manifest hash,
host, Python version and Slurm ID. Completed tasks are skipped only with the
same manifest. Failed attempts remain intact and can be retried into a new
attempt directory. A forcibly killed process can leave a `state/.../RUNNING`
lock; remove only that empty lock after confirming no worker remains. Changing
analysis settings requires a new manifest/output root.

Primary format references: [Relate modules](https://myersgroup.github.io/relate/modules.html),
[CLUES2 source](https://github.com/avaughn271/CLUES2).

CPU checks (no native Relate execution):

```bash
pytest --confcutdir=tests/rerun tests/rerun/test_relate_clues.py -q
```
