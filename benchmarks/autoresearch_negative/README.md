# Negative-selection autoresearch

Fit the frozen GRK2–GIH and TREM2–IBS genomic Gamma TMRCA targets with explicit
recurrent deleterious mutations in SLiM. This is a separate model family and
campaign, inspired by https://github.com/karpathy/autoresearch. It does not turn
msprime's sweep coefficient negative or rescale a fitted curve.

`prepare.py` freezes Ensembl Havana 104 exons union UCSC hg38 phastCons elements
within the existing 3-Mb regions. `simulate.py` uses either those annotations or
an explicitly synthetic contiguous selected region. Inside selected intervals,
a fitted fraction of new mutations has a fixed homozygous fitness effect s < 0;
the rest are neutral. Outside the intervals mutations are neutral. h = 0.5 is
fixed. Genotype fitnesses are 1, 1 + hs, and 1 + s. There is no beneficial
mutation, conditioned fixation, migration, or sweep endpoint.

The total mutation rate is the same 1.25e-8 per bp per generation as the empirical
calibration. The deCODE GRCh38 recombination map, 3-Mb context, central 960-kb
window, 20-kb bins, sample size and historical 20 pair indices stay fixed. The
Gamma prior is the original chromosome calibration, not a regional re-estimate.
The evaluator is the unchanged weighted log RMSE: 75% across all bins and 25%
across the central 200 kb. All bins are fitted; validation uses fresh random seeds.

The primary genotype assay retains actual selected mutations and overlays only
the appropriate neutral mutation fraction, using stdpopsim's SLiM processing.
Only biallelic polymorphisms are passed to Gamma; excluded counts are recorded.
Each replicate also reports true genealogical TMRCA and a separate independent
neutral-marker Gamma assay on the same genealogy. A low inferred TMRCA caused
mainly by depletion of selected variants must not be described as an equally
deep genealogical trough.

Initial fitting uses SLiM rescaling Q = 10 and a 10Ne forward burn-in. Scaling
is approximate: population sizes are divided by Q, selection and mutation rates
are multiplied by Q, recombination is transformed by stdpopsim, and returned
times are converted to physical generations. Record Q in every figure. Never
claim these runs are equivalent to unscaled Q = 1. Follow-up checks at smaller Q
and longer burn-in quantify sensitivity, not biological identifiability.

The first campaign fits 20 configurations per locus (2 neutral controls and 18
negative-selection candidates) with 3 common fitting seeds each. It freezes the
best negative configuration before evaluating 24 new seeds. Neutral controls are
scored but cannot be selected as the requested best negative model. Synthetic
block width, position and deleterious fraction are descriptive fit parameters;
they are not functional annotations. Negative selection can fail to fit the
observations. All failures and unsuccessful proposals remain in the history.

Use two low-priority CPU workers per stage and the existing shared GPU inference
lock. Independent locus fitting and frozen-winner checks may overlap using four
CPU workers in total; leave other tasks' processes alone. Launch from a new frozen source directory.
Timeout cleanup terminates the entire owned subprocess group, including SLiM.
Final arrays and plot metadata preserve provenance hashes. Large tree files stay
on the execution host.

`plot.py --live` renders completed validation for finished loci and the best
completed fitting ensemble for unfinished loci, with explicit stage and replicate
labels. It snapshots every displayed candidate/neutral ensemble and search
history, recomputes the displayed scores, and checks them against the source.
Use a new output directory per render to preserve earlier figure evidence.
Without `--live`, all loci must have completed validation summaries.

References: [SLiM](https://messerlab.org/slim/),
[pyslim workflow](https://tskit.dev/pyslim/docs/stable/tutorial.html),
[stdpopsim](https://popsim-consortium.github.io/stdpopsim-docs/stable/).
