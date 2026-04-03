"""
Fast demographic flow field with log-space numerics.
"""
import numpy as np
from scipy.special import digamma, gammaln, gammaincc
from scipy.stats import gamma as gamma_dist
import tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmrca_cu import _core


def _build_hazard(Ne_values, epoch_boundaries, Ne_ref, t_grid):
    M = len(Ne_values)
    scale = 2.0 * Ne_ref
    t_bounds = np.array(epoch_boundaries[:M+1]) / scale
    t_bounds[0] = 0.0
    lam_vals = Ne_ref / np.array(Ne_values, dtype=np.float64)

    cum_at_bound = np.zeros(M + 1)
    for k in range(1, M):
        cum_at_bound[k] = cum_at_bound[k-1] + lam_vals[k-1] * (t_bounds[k] - t_bounds[k-1])

    lam_arr = np.empty_like(t_grid)
    cum_arr = np.empty_like(t_grid)
    for idx, t in enumerate(t_grid):
        k = M - 1
        for kk in range(M - 1):
            if t < t_bounds[kk + 1]:
                k = kk
                break
        lam_arr[idx] = lam_vals[k]
        cum_arr[idx] = cum_at_bound[k] + lam_vals[k] * (t - t_bounds[k])

    return lam_arr, cum_arr


def generate_flow_field(Ne_values, epoch_boundaries, Ne_ref,
                         mean_min=1e-5, mean_max=1e2,
                         cv_min=1e-2, cv_max=1.0,
                         mean_n=51, cv_n=50, n_t=500, verbose=True):
    t_max = 80.0
    t_grid = np.linspace(1e-8, t_max, n_t)
    dt = t_grid[1] - t_grid[0]
    sqrt_dt = np.sqrt(dt)
    log_t = np.log(np.maximum(t_grid, 1e-30))

    lam_t, cum_lam_t = _build_hazard(Ne_values, epoch_boundaries, Ne_ref, t_grid)
    log_lam_t = np.log(np.maximum(lam_t, 1e-30))

    log10_mean = np.linspace(np.log10(mean_min), np.log10(mean_max), mean_n)
    log10_cv = np.linspace(np.log10(cv_min), np.log10(cv_max), cv_n)
    ln10 = np.log(10.0)

    u_out = np.zeros((mean_n, cv_n))
    v_out = np.zeros((mean_n, cv_n))

    for i in range(mean_n):
        for j in range(cv_n):
            mean = 10.0**log10_mean[i]
            cv = 10.0**log10_cv[j]
            std = cv * mean
            alpha = (mean / std)**2
            beta = mean / std**2

            if alpha < 1e-6 or beta < 1e-6 or alpha > 1e6:
                continue

            # Log-space Gamma PDF: log(f(t)) = alpha*log(beta) - gammaln(alpha)
            #                                  + (alpha-1)*log(t) - beta*t
            log_gamma_pdf = (alpha * np.log(beta) - gammaln(alpha)
                             + (alpha - 1) * log_t - beta * t_grid)
            gamma_pdf = np.exp(log_gamma_pdf)

            # Gamma SF: P(T_old > t) = gammaincc(alpha, beta*t) (regularized upper incomplete)
            gamma_sf = gammaincc(alpha, beta * t_grid)

            # ── SMC' transition ──────────────────────────────
            # Term 1: lambda(t) * exp(-Lambda(t)) * P(T_old > t)
            log_term1 = log_lam_t - cum_lam_t + np.log(np.maximum(gamma_sf, 1e-300))
            term1 = np.exp(log_term1)

            # Term 2: 2*lambda(t) * exp(-2*Lambda(t)) * cumint
            # where cumint(t) = integral_0^t exp(Lambda(s)) * Gamma_pdf(s) ds
            #
            # Work in log space: log(exp(Lambda(s)) * Gamma_pdf(s))
            #                  = Lambda(s) + log_gamma_pdf(s)
            log_integrand = cum_lam_t + log_gamma_pdf

            # Stabilize: subtract max for numerical safety
            log_max = np.max(log_integrand)
            if np.isfinite(log_max):
                integrand_stable = np.exp(log_integrand - log_max)
                cumint_stable = np.cumsum(integrand_stable) * dt
                # term2 = 2 * lambda(t) * exp(-2*Lambda(t)) * exp(log_max) * cumint_stable
                log_term2_base = np.log(2.0) + log_lam_t - 2.0 * cum_lam_t + log_max
                # Where cumint_stable > 0
                safe_mask = cumint_stable > 0
                term2 = np.zeros_like(t_grid)
                term2[safe_mask] = np.exp(
                    log_term2_base[safe_mask] + np.log(cumint_stable[safe_mask]))
            else:
                term2 = np.zeros_like(t_grid)

            f_new = term1 + term2
            delta = (f_new - gamma_pdf) * sqrt_dt

            # Gamma partial derivatives
            d_alpha = gamma_pdf * (np.log(beta) - digamma(alpha) + log_t) * sqrt_dt
            d_beta = gamma_pdf * (alpha / beta - t_grid) * sqrt_dt
            d_la = d_alpha * alpha / ln10
            d_lb = d_beta * beta / ln10

            A = np.column_stack([d_la, d_lb])
            # Only use points where Gamma PDF is non-negligible
            wt = gamma_pdf > gamma_pdf.max() * 1e-10
            if wt.sum() < 3:
                continue

            result, _, _, _ = np.linalg.lstsq(A[wt], delta[wt], rcond=None)
            c1, c2 = result

            u_out[i, j] = c1 - c2
            v_out[i, j] = -0.5 * c1

        if verbose and ((i + 1) % 10 == 0 or i == mean_n - 1):
            print(f"  flow field: {(i+1)*cv_n}/{mean_n*cv_n}", flush=True)

    return u_out, v_out


def write_flow_field(path, u, v, mean_min=1e-5, mean_max=1e2,
                      cv_min=1e-2, cv_max=1.0):
    mean_n, cv_n = u.shape
    with open(path, 'w') as f:
        f.write(f"{mean_min} {mean_max} {mean_n}\n")
        f.write(f"{cv_min} {cv_max} {cv_n}\n")
        for i in range(mean_n):
            f.write(' '.join(f"{u[i,j]:.10f}" for j in range(cv_n)) + '\n')
        for i in range(mean_n):
            f.write(' '.join(f"{v[i,j]:.10f}" for j in range(cv_n)) + '\n')


def load_flow_field(path):
    with open(path) as f:
        parts = f.read().split()
    idx = 0
    mean_min = float(parts[idx]); mean_max = float(parts[idx+1]); mean_n = int(parts[idx+2]); idx += 3
    cv_min = float(parts[idx]); cv_max = float(parts[idx+1]); cv_n = int(parts[idx+2]); idx += 3
    u = np.array([float(x) for x in parts[idx:idx+mean_n*cv_n]]).reshape(mean_n, cv_n); idx += mean_n*cv_n
    v = np.array([float(x) for x in parts[idx:idx+mean_n*cv_n]]).reshape(mean_n, cv_n)
    return u, v, mean_min, mean_max, cv_min, cv_max


def estimate_demography(tmrca_means, Ne_init, n_epochs=20, t_max=None):
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


class DemographicFlowContext:
    """Two-stage demographic-aware TMRCA estimation."""

    def __init__(self, G, positions, Ne_init, mu, rho, flow_field_path,
                 n_calibration_pairs=50, n_epochs=20, n_t=500,
                 cache_steps=0, gpu_id=0, verbose=True):
        self.S = len(positions)
        self.Ne_init = Ne_init
        n = G.shape[0]

        _core.set_device(gpu_id)

        if verbose:
            print("Stage 1: estimating demography...", flush=True)

        rng = np.random.default_rng(42)
        n_cal = min(n_calibration_pairs, n * (n - 1) // 2)
        cal_set = set()
        while len(cal_set) < n_cal:
            a, b = sorted(rng.choice(n, 2, replace=False))
            cal_set.add((int(a), int(b)))

        cal_ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho,
                                     flow_field_path, cache_steps)
        cal_result = cal_ctx.run_fb(sorted(cal_set), mean_only=True)
        del cal_ctx

        self.Ne_values, self.epoch_boundaries = estimate_demography(
            cal_result["mean"], Ne_init, n_epochs=n_epochs)

        if verbose:
            print(f"  N(t): {self.Ne_values[:3].astype(int)} ... "
                  f"{self.Ne_values[-2:].astype(int)}", flush=True)
            print("Stage 2: computing flow field...", flush=True)

        import time
        t0 = time.perf_counter()
        u, v = generate_flow_field(
            self.Ne_values, self.epoch_boundaries, Ne_init,
            n_t=n_t, verbose=verbose)
        if verbose:
            print(f"  flow field in {time.perf_counter()-t0:.1f}s", flush=True)

        self._ff_dir = tempfile.mkdtemp()
        self._ff_path = os.path.join(self._ff_dir, "demographic_flow_field.txt")
        write_flow_field(self._ff_path, u, v)

        self.ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho,
                                      self._ff_path, cache_steps)
        if verbose:
            print("Ready.", flush=True)

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
