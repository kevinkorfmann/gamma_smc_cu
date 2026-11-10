# Autoresearch-style genomic profile matching

Exploratory parameter optimization for GRK2–GIH and TREM2–IBS TMRCA profiles.
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):
freeze evaluation, establish a baseline, propose a change, run a bounded
experiment, retain improvements, and record every outcome. This is a standalone
adaptation, not a dependency on the LLM-training repository. The parameter
proposals are automated numerical search; the supervising agent reviews results
and may propose another separately recorded campaign.

`prepare.py` checks archived `COMPLETE.json` hashes and physical-time metadata,
converts VCF POS to zero-based coordinates, freezes the observed 20-pair Gamma
profiles in 20-kb bins, and snapshots the matching deCODE GRCh38 recombination
map. YRI profiles are retained as descriptive controls. The main targets are
the historical pair subsamples, not exhaustive population averages.

`simulate.py` uses msprime's conditional stochastic sweep ancestry model,
neutral mutation overlay, and the same population sample size, pair indices,
physical mutation rate, scalar inference recombination rate, and fixed
chromosome-calibrated Gamma prior as the observations. The generating map stays
fixed. It records both genealogical truth and Gamma inference. A 3-Mb simulation
contains the central 960-kb evaluation window with roughly 1-Mb buffers. The
observations used full-chromosome inference; this residual context mismatch
requires sensitivity checks before substantive interpretation.

`metrics.py` is frozen per campaign. The objective is a weighted log RMSE of
the **replicate-mean inferred profile**: 75% over the scoring bins and 25% over
the central 200 kb. There is no post-hoc amplitude fitting, horizontal shift,
target smoothing, flexible recombination-map fitting, or best-seed selection.
Alternating 80-kb blocks are withheld during optimization. Spatial blocks are
correlated; independent random-seed replication is the stronger validation.
Failed or incomplete curves cannot gain an advantage by dropping difficult bins.

`search.py` varies constant diploid Ne (3,000–40,000), sweep coefficient
(0.001–0.3), time since the sweep's endpoint (0–5,000 generations), selected
position (within 150 kb of the gene center), endpoint frequency (0.5–1), and
hard versus standing-variation starting frequencies (0.001–0.15 for standing).
It begins with a neutral baseline and prespecified sweep hypotheses, then
alternates broad proposals and local refinements. Three shared training seed
identities per candidate reduce simulation noise; no seed is optimized. The
selected configuration is frozen before comparison with neutral on 12 new
replicates. All candidate results and crashes survive; restart requires identical
inputs, code, settings, API, flow field and compiled library hashes.

These are conditional resemblance experiments. They do not establish positive
selection, provide ABC/posterior probabilities, estimate establishment rates,
or rule out BGS. There is no mechanistic BGS, population-specific demography,
empirical accessibility masking, or phasing error in this first model class.
The selection coefficient follows msprime's additive genic convention. Do not
silently interpret these fitted parameters as biological estimates. Independent
SLiM confirmation and context/demography sensitivity are subsequent work.

An optional `--initial-configs` JSON maps each target key to candidate settings
from a previous, completed campaign. This provides an explicit record of the
supervising agent's hypotheses. A candidate may include `ancient_Ne` and
`change_time` (generations before present) to specify a two-epoch demographic
model. Demographic events falling inside msprime's unsupported sweep phase fail
visibly; they are never silently ignored or shifted. Use distinct seeds and a
new frozen source/output directory for each refinement campaign. Its fresh-seed
validation must not reuse a previous campaign's validation seeds.

Example (set paths to the isolated run directory and validated Gamma environment):

```sh
python prepare.py --source empirical --out targets
python -m unittest discover -s . -p 'test_*.py'
python search.py --targets targets --out campaign01 --api frozen_gamma_api.py \
  --gamma-lock /path/to/shared/gamma.lock --candidates 40 --replicates 3 \
  --validation-replicates 12
python plot.py --targets targets --run campaign01 --out figures01
```

The two-row plot uses a logarithmic TMRCA axis, the manuscript's population
colours, focal-gene shading and a shared model legend. Its lower row places the
final search trace beside the retained configuration: generating diploid Ne,
genic selection coefficient, time since the sweep endpoint, selected coordinate,
offset, effective frequency endpoints, sample size and fixed inference Ne.
The hard-sweep start is one copy, not the inactive stored `start_frequency=0`.
Coordinates in the parameter card are one-based GRCh38. Means and 10–90%
replicate ribbons retain the frozen scoring statistic; they are not the paper's
pairwise median/IQR. `figure_metadata.json` records full-precision parameters,
input/output hashes and the plotted-array checks. Optional `--fonts DIR` loads
manuscript TrueType fonts. Render into a new output directory to preserve earlier
presentations and leave frozen campaign sources unchanged.

Use an explicitly assigned GPU and the existing shared inference lock. Each
replicate has a 300-second subprocess timeout; candidates have identical replicate
counts. The numeric evaluator is not modified after seeing fit results. Use a
new campaign directory for a new hypothesis or budget; never overwrite evidence.
`--workers` permits up to four independent CPU simulations; the shared lock
still serializes all GPU inference calls. Replicate collection preserves seed order.

For literal whole-curve reproduction, `--fit-bins all_bins` fits every observed
bin and relies on independent simulation seeds for validation. This is a
separate, explicitly recorded selection protocol; there is then **no spatial
holdout** (the legacy subset scores remain descriptive only). It avoids omitting
unique flank structure from fitting. Never claim improvements in a training
metric across these two protocols as though the selection objective were identical.
