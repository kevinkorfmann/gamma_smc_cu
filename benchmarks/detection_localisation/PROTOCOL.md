# Controlled detection and localisation benchmark

Protocol fixed before production results, 2026-09-27. This is a new controlled
experiment, separate from empirical locus-fitting. It tests performance in
specified hard-sweep regimes and does not calibrate the human atlas or resolve
background-selection confounding. No empirical gene is used to choose settings.

## Generating model

SLiM 4.3 via stdpopsim 0.3.0, with diploid Ne=10,000, two daughter populations
split 6,000 generations ago from an ancestor of the same size, no migration,
mu=1.25e-8, 1-Mb sequence, and uniform recombination 1e-8 per bp per generation.
Sample 50 diploid individuals in each population. Use Q=1 (no rescaling) and
0.1 Ne forward burn-in before the split, with neutral ancestral recapitation.
One beneficial mutation is introduced into the focal population at a position
sampled uniformly within the central 200 kb; fitnesses 1, 1+s/2, 1+s, s=0.02.
Neutral mutations are added once by stdpopsim. Exclude the known causal site
from ALL inference inputs, but retain its realised population/sample frequency
and trajectory. All other biallelic pooled-segregating variants enter XP-EHH;
focal segregating variants enter Gamma; focal MAF>=0.05 variants enter iHS.
This is a neutral-marker ascertainment benchmark with known generating rates.

Prespecified regimes (times in generations before sampling):
- Neutral, no beneficial mutation.
- Incomplete: mutation introduced 800 ago; final population frequency 0.25–0.75.
- Near fixation: introduced 1,200 ago; final frequency 0.80–0.99.
- Completed: introduced 2,000 ago; fixed by sampling.
- Older completed: introduced 4,000 ago; fixed at least 2,000 generations ago.

Frequency conditioning uses rejection sampling; all attempts/timeouts are
recorded. Estimates are conditional on these histories, not establishment
probabilities. Actual fixation ages are reported; introduction age is not
presented as fixation age. A small technical pilot validates implementation
and resource use, and is excluded from final estimates. Protocol amendments
must precede production and record the reason independently of method rankings.

## Calibration, scoring and evaluation

Separate independent simulation IDs for score normalisation, threshold
calibration and evaluation. Target 200 neutral normalisation replicates,
200 neutral threshold-calibration replicates, 200 neutral evaluation replicates,
and 100 evaluation replicates per selected regime. Seeds are deterministic
hashes of the protocol ID, split, regime and replicate index. This initial
scope is one demographic/recombination setting, not general power across
human ancestries or sweep types.

Fresh iHS and XP-EHH are computed on the same simulated sequences using
scikit-allel, min EHH=0.05, gap scale=20 kb, max gap=200 kb, incomplete edge
scores missing. iHS standardisation uses 20 derived-frequency bins from the
independent neutral normalisation set, with at least 50 finite scores per bin;
XP-EHH uses its pooled neutral mean and SD. Do not normalise each selected
replicate separately. Gamma uses the known Ne, mu and scalar recombination
rate, all 4,950 pairs among the 100 focal haplotypes, full 1-Mb context, and
mean log posterior-mean TMRCA across pairs/sites (the atlas aggregation).

For each independent replicate, report scores in a 5-kb and 20-kb interval
centred on the known selected position (a pseudo-target in neutrals). Require
at least two focal segregating sites for Gamma. Primary haplotype summaries
are fractions |iHS_z|>2 or XP-EHH_z>2 among finite core scores. Regional iHS
and XP-EHH summaries use disjoint 100-kb flanks. Maximum-score summaries are
secondary checks. Missing scores remain missing and count as no detection
in unconditional power; conditional-on-computability power is separate.

Choose each method/interval threshold from independent neutral calibration
at a nominal 5% target-interval false-positive rate. Use strict empirical
tails to handle ties conservatively; report achieved held-out neutral FPR,
not only nominal FPR. Report independent-replicate Wilson 95% intervals for
power, coverage and FPR; paired bootstrap intervals for method differences.

For localisation, scan a fixed central 500-kb interval using 5-kb and 20-kb
windows stepped by 5 kb. Apply the same summary to every window. Report the
absolute distance between the known site and the strongest window centre,
including all ties (median distance across tied best windows, tie span and
frequency separately). This is localisation of a simulated causal position
by window scores, not proof of causal-gene identification. Report localisation
for all computable selected replicates and separately among detected
replicates; selection on detection must be explicit. Scan-level detection
uses separately calibrated neutral scan-maximum thresholds, controlling the
whole evaluated interval. No threshold/window/score is chosen on test outcomes.

## Required verification

Check population/sample identities, time units, causal-site removal, unique
positions, site/score denominators, exact input sharing, no overlapping split
seeds, Gamma time calibration and XP-EHH sign under population exchange.
Verify interval coordinates, tied thresholds, missing-score treatment and
localisation against synthetic examples. Preserve manifests, generated SLiM
scripts, accepted trajectories, input hashes, software versions, failures and
per-replicate results. Run all simulations/inference through Slurm on Betty.

Sources: https://popsim-consortium.github.io/stdpopsim-docs/stable/api.html ;
https://tskit.dev/pyslim/docs/stable/tutorial.html ;
https://scikit-allel.readthedocs.io/en/stable/stats/selection.html

## Execution amendment before production

The five simulation pilots passed without changing the biological scenarios.
GPU capacity was unavailable at scoring dispatch. The production Gamma score
will therefore use the upstream CPU Gamma-SMC reference, with explicit theta
4Ne*mu=0.0005 and rho=4Ne*r=0.0004 and posterior means converted by 2Ne=20,000.
This evaluates the Gamma-SMC statistic; it is not a new GPU speed comparison.
The same five inputs are queued for CUDA scoring as a numerical sensitivity
check. The backend and source/binary/flow hashes are recorded, and the paper
must identify the CPU reference if that is the production backend. No method
performance outcomes were inspected to choose the backend or scenarios.
