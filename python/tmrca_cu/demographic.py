"""
Demographic correction for the Gamma-SMC flow field pipeline.

Two-stage pipeline:
  1. Estimate piecewise-constant N(t) from a subsample of pairs
  2. Scale the default flow field to account for the demographic prior
  3. Run all pairs with the corrected flow field

The key insight: the flow field vectors encode the displacement toward
the coalescent prior per unit of recombination. Under variable N(t),
the prior changes — the mean coalescent time and its variance shift.
We rescale the flow field vectors to match the demographic prior's
moments, using the default constant-Ne flow field as a template.

Usage:
    from tmrca_cu.demographic import DemographicFlowContext

    ctx = DemographicFlowContext(G, positions, Ne_init=10000, mu=1.25e-8, rho=1e-8,
                                 flow_field_path=FF, n_calibration_pairs=50)
    result = ctx.run_fb(all_pairs, mean_only=True)
"""
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmrca_cu import _core


# ══════════════════════════════════════════════════════════════
# Demographic coalescent prior: piecewise-constant N(t)
# ══════════════════════════════════════════════════════════════

def _coalescent_prior_moments(Ne_values, epoch_boundaries):
    """Compute E[T] and E[T^2] of coalescent time under piecewise-constant N(t).

    Parameters
    ----------
    Ne_values : array, shape (M,)
        Diploid effective population size in each epoch.
    epoch_boundaries : array, shape (M+1,)
        Time boundaries [0, t1, t2, ..., inf].

    Returns
    -------
    mean : float  — E[T] in generations
    var : float   — Var[T] in generations^2
    """
    M = len(Ne_values)
    lam = 1.0 / (2.0 * np.array(Ne_values, dtype=np.float64))
    t = np.array(epoch_boundaries[:M+1], dtype=np.float64)
    t[0] = 0.0

    cum_lam = np.zeros(M + 1)
    for k in range(1, M):
        cum_lam[k] = cum_lam[k-1] + lam[k-1] * (t[k] - t[k-1])

    mean = 0.0
    mean2 = 0.0

    for k in range(M):
        lk = lam[k]
        a = t[k]
        b = t[k+1] if k < M - 1 else np.inf
        S_a = np.exp(-cum_lam[k])
        if S_a < 1e-300:
            break

        if b == np.inf:
            mean += S_a * (a + 1.0 / lk)
            mean2 += S_a * (a**2 + 2.0 * a / lk + 2.0 / lk**2)
        else:
            dt = b - a
            exp_neg = np.exp(-lk * dt)
            mean += S_a * (a * (1 - exp_neg) + (1.0/lk) * (1 - exp_neg) - dt * exp_neg)
            x = lk * dt
            ex = exp_neg
            mean2 += S_a * (
                a**2 * (1 - ex) +
                2 * a * ((1.0/lk) * (1 - ex) - dt * ex) +
                (2.0/lk**2) * (1 - ex) - (2.0 * dt / lk) * ex - dt**2 * ex
            )

    var = mean2 - mean**2
    return mean, var


# ══════════════════════════════════════════════════════════════
# Flow field rescaling
# ══════════════════════════════════════════════════════════════

def load_flow_field(path):
    """Load a flow field from Schweiger's text format."""
    with open(path) as f:
        parts = f.read().split()
    idx = 0
    mean_min = float(parts[idx]); mean_max = float(parts[idx+1]); mean_n = int(parts[idx+2]); idx += 3
    cv_min = float(parts[idx]); cv_max = float(parts[idx+1]); cv_n = int(parts[idx+2]); idx += 3
    u = np.array([float(x) for x in parts[idx:idx+mean_n*cv_n]]).reshape(mean_n, cv_n); idx += mean_n*cv_n
    v = np.array([float(x) for x in parts[idx:idx+mean_n*cv_n]]).reshape(mean_n, cv_n)
    return u, v, mean_min, mean_max, cv_min, cv_max


def rescale_flow_field(u_orig, v_orig, Ne_const, Ne_values, epoch_boundaries,
                        mean_min=1e-5, mean_max=1e2, cv_min=1e-2, cv_max=1.0):
    """Rescale a constant-Ne flow field for a demographic model.

    The flow field vectors point from each grid point toward the coalescent
    prior. Under variable N(t), the prior's location in (mean, CV) space
    shifts. We rescale the vectors by the ratio of demographic-to-constant
    prior moments, adjusted per grid point based on how far it is from the
    prior.

    Parameters
    ----------
    u_orig, v_orig : ndarray, shape (mean_n, cv_n)
        Original constant-Ne flow field.
    Ne_const : float
        The Ne used for the original flow field.
    Ne_values, epoch_boundaries : array-like
        Piecewise-constant demographic model.

    Returns
    -------
    u_new, v_new : ndarray
        Rescaled flow field.
    """
    # Constant-Ne prior: E[T] = 2*Ne (in generations), = 1 (in coalescent units)
    # CV of exponential = 1, so log10(mean_prior) = 0, log10(CV_prior) = 0

    # Demographic prior moments
    mean_demo_gen, var_demo_gen = _coalescent_prior_moments(Ne_values, epoch_boundaries)
    # Convert to coalescent units (relative to Ne_const)
    mean_demo = mean_demo_gen / (2.0 * Ne_const)
    cv_demo = np.sqrt(var_demo_gen) / mean_demo_gen  # CV is unitless

    # Constant-Ne prior: mean = 1.0 (coalescent units), CV = 1.0
    mean_const = 1.0
    cv_const = 1.0

    # The flow field vectors at each grid point push toward the prior.
    # The u-component (mean direction) scales with how far the prior mean shifted.
    # The v-component (CV direction) scales with how far the prior CV shifted.
    mean_n, cv_n = u_orig.shape
    log10_mean_grid = np.linspace(np.log10(mean_min), np.log10(mean_max), mean_n)

    # Scale factor for mean direction: the prior is now at log10(mean_demo) instead of 0
    # For each grid point, the displacement toward the prior changes proportionally
    log10_mean_prior_const = np.log10(mean_const)  # = 0
    log10_mean_prior_demo = np.log10(mean_demo)
    log10_cv_prior_const = np.log10(cv_const)      # = 0
    log10_cv_prior_demo = np.log10(cv_demo)

    u_new = u_orig.copy()
    v_new = v_orig.copy()

    for i, lm in enumerate(log10_mean_grid):
        # Distance from grid point to constant prior (in mean direction)
        d_const = log10_mean_prior_const - lm
        d_demo = log10_mean_prior_demo - lm

        # Scale the u-vector by the ratio of distances
        if abs(d_const) > 1e-6:
            scale_u = d_demo / d_const
        else:
            scale_u = 1.0

        u_new[i, :] *= scale_u

    # Similar for CV direction — but the prior CV shift is usually smaller
    log10_cv_grid = np.linspace(np.log10(cv_min), np.log10(cv_max), cv_n)
    for j, lc in enumerate(log10_cv_grid):
        d_const = log10_cv_prior_const - lc
        d_demo = log10_cv_prior_demo - lc
        if abs(d_const) > 1e-6:
            scale_v = d_demo / d_const
        else:
            scale_v = 1.0
        v_new[:, j] *= scale_v

    return u_new, v_new


def write_flow_field(path, u, v,
                      mean_min=1e-5, mean_max=1e2,
                      cv_min=1e-2, cv_max=1.0):
    """Write flow field arrays to Schweiger's text format."""
    mean_n, cv_n = u.shape
    with open(path, 'w') as f:
        f.write(f"{mean_min} {mean_max} {mean_n}\n")
        f.write(f"{cv_min} {cv_max} {cv_n}\n")
        for i in range(mean_n):
            f.write(' '.join(f"{u[i,j]:.10f}" for j in range(cv_n)) + '\n')
        for i in range(mean_n):
            f.write(' '.join(f"{v[i,j]:.10f}" for j in range(cv_n)) + '\n')


# ══════════════════════════════════════════════════════════════
# Demographic estimation from TMRCA posteriors
# ══════════════════════════════════════════════════════════════

def estimate_demography(tmrca_means, Ne_init, n_epochs=20, t_max=None):
    """Estimate piecewise-constant N(t) from TMRCA posterior means.

    Parameters
    ----------
    tmrca_means : ndarray
        Posterior mean TMRCA in generations.
    Ne_init : float
        Initial Ne estimate.
    n_epochs : int
        Number of piecewise-constant epochs.

    Returns
    -------
    Ne_values : ndarray, shape (n_epochs,)
    epoch_boundaries : ndarray, shape (n_epochs + 1,)
    """
    if t_max is None:
        t_max = 10.0 * Ne_init

    vals = np.asarray(tmrca_means).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]

    fracs = np.linspace(0, 1, n_epochs + 1)**2
    boundaries = t_max * fracs
    boundaries[0] = 0.0

    Ne_values = np.full(n_epochs, Ne_init, dtype=np.float64)

    for k in range(n_epochs):
        lo, hi = boundaries[k], boundaries[k + 1]
        mask = (vals >= lo) & (vals < hi)
        count = mask.sum()
        if count < 10:
            continue
        dt = hi - lo
        if dt > 0:
            density = count / (len(vals) * dt)
            survival = np.mean(vals >= lo)
            if survival > 0.01:
                lam = density / survival
                Ne_values[k] = max(100.0, 1.0 / (2.0 * lam))

    return Ne_values, boundaries


# ══════════════════════════════════════════════════════════════
# DemographicFlowContext: two-stage pipeline
# ══════════════════════════════════════════════════════════════

class DemographicFlowContext:
    """Two-stage demographic-aware TMRCA estimation.

    Stage 1: Run a subsample with the default constant-Ne flow field,
             estimate N(t), rescale the flow field.
    Stage 2: Run all pairs with the corrected flow field.

    Parameters
    ----------
    G : ndarray, shape (n, S), dtype uint8
    positions : ndarray, shape (S,), dtype float64
    Ne_init : float
        Initial effective population size estimate.
    mu, rho : float
        Per-site per-generation mutation and recombination rates.
    flow_field_path : str
        Path to default (constant-Ne) flow field for calibration stage.
    n_calibration_pairs : int
        Number of pairs for demographic estimation.
    n_epochs : int
        Number of piecewise-constant epochs for N(t).
    cache_steps : int
        Flow field cache depth (0 = auto).
    gpu_id : int
        GPU device.
    """

    def __init__(self, G, positions, Ne_init, mu, rho, flow_field_path,
                 n_calibration_pairs=50, n_epochs=20, cache_steps=0, gpu_id=0):
        self.G = G
        self.positions = positions
        self.Ne_init = Ne_init
        self.mu = mu
        self.rho = rho
        self.S = len(positions)
        n = G.shape[0]

        _core.set_device(gpu_id)

        # ── Stage 1: calibration ──────────────────────────────
        rng = np.random.default_rng(42)
        n_cal = min(n_calibration_pairs, n * (n - 1) // 2)
        cal_set = set()
        while len(cal_set) < n_cal:
            a, b = sorted(rng.choice(n, 2, replace=False))
            cal_set.add((int(a), int(b)))
        cal_pairs = sorted(cal_set)

        cal_ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho,
                                     flow_field_path, cache_steps)
        cal_result = cal_ctx.run_fb(cal_pairs, mean_only=True)
        cal_means = cal_result["mean"]
        del cal_ctx

        # Estimate N(t)
        self.Ne_values, self.epoch_boundaries = estimate_demography(
            cal_means, Ne_init, n_epochs=n_epochs)

        # ── Stage 2: rescale flow field ───────────────────────
        u_orig, v_orig, mean_min, mean_max, cv_min, cv_max = load_flow_field(flow_field_path)

        u_new, v_new = rescale_flow_field(
            u_orig, v_orig, Ne_init,
            self.Ne_values, self.epoch_boundaries,
            mean_min, mean_max, cv_min, cv_max)

        self._ff_dir = tempfile.mkdtemp()
        self._ff_path = os.path.join(self._ff_dir, "demographic_flow_field.txt")
        write_flow_field(self._ff_path, u_new, v_new, mean_min, mean_max, cv_min, cv_max)

        self.ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho,
                                      self._ff_path, cache_steps)

    def run_fb(self, pairs, mean_only=True):
        return self.ctx.run_fb(pairs, mean_only)

    def run_fwd(self, pairs, mean_only=True):
        return self.ctx.run_fwd(pairs, mean_only)

    def run_fb_summary(self, pairs):
        return self.ctx.run_fb_summary(pairs)

    @property
    def device_id(self):
        return self.ctx.device_id

    @property
    def demography(self):
        return self.Ne_values.copy(), self.epoch_boundaries.copy()

    def __del__(self):
        try:
            if hasattr(self, '_ff_path') and os.path.exists(self._ff_path):
                os.unlink(self._ff_path)
            if hasattr(self, '_ff_dir') and os.path.exists(self._ff_dir):
                os.rmdir(self._ff_dir)
        except Exception:
            pass
