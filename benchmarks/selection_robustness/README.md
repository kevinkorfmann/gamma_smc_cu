# Selection robustness: technical benchmark components

Experimental scripts for SLiM simulations with real human recombination maps,
deleterious mutations on exon annotations, and focal directional or balancing
selection. These scripts currently implement a **technical smoke test**, not
a validated publication analysis. The simulator requires `--technical-smoke`.

- `prepare_region.py`: snapshot GRCh38 map, exon and B-map annotations with
  coordinate/coverage validation and provenance.
- `select_regions.py`: select map/B-stratified genomic tiles without observed
  TMRCA ranks, preserving a held-out chromosome split and exclusion counts.
- `simulate.py`: run SLiM, retain true genealogies, construct shared binary
  inputs/pairs and compute exact physical-window TMRCA truth.
- `prepare_methods.py`: generate constant-prior ASMC decoding quantities from
  independent neutral data and verify a pinned cxt checkpoint.
- `infer.py`: separate-process adapters for Gamma-SMC-CU, ASMC and cxt, with
  explicit input-map differences, raw predictions and aligned-window checks.
- `common.py`, `test_common.py`: interval-weighted scoring and coordinate tests.
- `abc_reference.py`, `test_abc_reference.py`: rejection-ABC calculation with
  declared model priors, training-only covariance whitening, accepted sample
  weights, independent held-out model confusion/Brier/reliability diagnostics,
  and tests against an analytically solvable model. This does not yet supply
  a biological reference bank, regional summary extraction or calibrated
  human parameter/model posteriors.

Simulation controls: neutral, BGS only, positive selection, positive plus BGS,
balancing selection, and balancing plus BGS. Recombination can be empirical or
uniform with equal total genetic length. The focal balancing implementation is
**asymmetric** overdominance (AA=1, Aa=1+s, aa=1-s), with deterministic equilibrium
frequency 1/3. Lost alleles are retained and labelled, not silently resimulated.
Published B maps stratify regions; they are not a substitute for mechanistic BGS.

### Separate B′ specificity experiment

`simulate_specificity.py` and `run_specificity_bank.py` implement a separate,
controlled TREM2-region experiment with an immutable job manifest and saved
population allele trajectories. The neutral/positive-only experiment uses
Ne=10,000, no rescaling, 2,000 forward generations and neutral recapitation
older than the entire selected phase. Every introduced mutation is retained,
including losses. This does not lift the production gates on `simulate.py`.

`bprime_refit.py` evaluates the deposited Buffalo–Kern initial YRI CADD6/deCODE
100-kb altgrid fit. For its single feature, beta=mu*W gives an exact convex
reparameterization of the authors' composite likelihood. It checks original
prediction/likelihood parity, analytical gradients and recovery from displaced
parameters. This reproduces the fit used by the current figure, not every fit
reported in the paper or its separate 1-Mb summary dataset.

`analyze_specificity.py` compares positive-only and neutral SLiM replicates,
excluding the selected site from diversity. It replaces expected pair counts
in 29 fully contained regional bins, using pi0*pi_sim/(4 Ne mu), and refits the
whole-genome model with all other data held fixed. This is a conditional
regional sensitivity experiment; it is not an empirical selection posterior,
a whole-genome forward simulation, or a comparison of TMRCA inference methods.
Independent-replicate bootstrap intervals, allele outcomes, optimizer-start
checks, data hashes and per-replicate results are retained. `plot_specificity.py`
exports PDF/PNG figures with pointwise intervals. The deCODE bank matches the
published predictor map; a Pyrho-YRI simulation bank tests sensitivity to the
generating map while retaining the same deCODE B′ predictor.

`analyze_specificity_scale.py` adds post-hoc averaging-window and calibration-
span diagnostics, reusing the original fixed-sweep/neutral cohort. It computes
exact interval statistics and validates the widest-span refits against the
original analysis. `plot_specificity_summary.py --scale ... --view full`
produces the six-panel overview, including the complete simulation interval,
global fitting-weight limitation, spatial averaging and allele outcomes.
`summarize_specificity.py` calculates ratios of ensemble means and their
replicate-bootstrap intervals. `validate_specificity_logger.py` verifies that
logging preserves every biological SLiM table in a same-seed replay.

The initial inference adapter supports one 1-Mb block, 50 haplotypes and a shared
pair list. Gamma-SMC-CU uses a scalar recombination rate; ASMC runs with either
the true map or its mean; cxt uses its native mutational context. cxt draw
averaging is performed after exponentiating log-TMRCA. SNP predictions use
nearest-site cells without extrapolation beyond first/last SNP, and incomplete
coverage is reported. Native posterior resolutions differ.

Requires SLiM 4.3, stdpopsim 0.3.0, msprime, pyslim, tskit, NumPy, pandas,
ASMC/PrepareDecoding, a CUDA Gamma-SMC-CU build, and cxt with compatible Torch
dependencies. Use an isolated Linux environment and set `PYTHONNOUSERSITE=1`
before installing or running. Keep sources under `study/` in a run root with
`env/`, `tools/`, `references/`, `regions/`, `runs/` and `logs/` siblings;
`run_python.sh` supplies a bounded execution environment for that layout.
Each CLI documents required inputs with `--help`. Outputs must use new paths.

Run the coordinate/truth tests with:

```bash
python -m unittest discover -s benchmarks/selection_robustness -p 'test_*.py'
```

For the ABC reference CLI, training and held-out NPZ files require `summaries`
(rows by features), `summary_names` (unique ordered strings), `model` and
`group_id` (one independent ancestral simulation per row). Model labels and
groups must be non-object string arrays. An optional observed NPZ contains
the identical `summary_names` and one `summaries` vector. Model priors are a
JSON object mapping labels to positive probabilities summing to one. Each
model's simulations must be draws from its declared parameter prior; adaptive
parameter proposals are unsupported. Store parameters separately keyed by
group ID, retaining inactive parameters as such. Accepted indices/weights
support model-conditional parameter densities, not unvalidated pooling across
models. Never multiply support from the three correlated inference methods.

```bash
python abc_reference.py --train training.npz --heldout test.npz \
  --model-prior model_prior.json --fraction 0.02 --out new_abc_result
```

Freeze tolerance and summaries using a separate calibration set before final
testing. Algorithm tests do not establish biological identifiability, parameter
interval coverage, or posterior predictive adequacy. Real-data interpretation
requires those additional checks and matched demographic, ascertainment,
sampling, masking and inference workflows.

Population-level allele trajectories, biological burn-in/rescaling validation,
buffered multi-block inference, named-gene aggregation and null-calibrated
detection must be completed before production analysis. Technical-pilot error
numbers should not be used to rank methods or rule out empirical confounding.

References: [stdpopsim catalog](https://popsim-consortium.github.io/stdpopsim-docs/stable/catalog.html),
[ASMC](https://github.com/PalamaraLab/ASMC), [cxt](https://github.com/kr-colab/cxt),
[B-map source](https://github.com/nwcol/bgs_lmr/tree/1d44ba129d354272c52aee705987cd2e6abd8a97).
