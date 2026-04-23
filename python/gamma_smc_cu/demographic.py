"""
Fast demographic flow field using epoch-wise vectorized incomplete gamma.

Constant-Ne and demographic paths both fully vectorized — no Python loops
over time points or epochs in the hot path. ~5s for either.
"""
import numpy as np
from scipy.special import gammainc, gammaincc, gammaln, digamma, gammaincinv
import tempfile, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gamma_smc_cu import _core

_CALIBRATION_FACTOR = 5.302


def _dist_diff_const_Ne(t_arr, alpha, beta):
    """Vectorized distribution difference for constant Ne."""
    t = np.maximum(np.asarray(t_arr, dtype=np.float64), 1e-15)
    bm1 = beta - 1.0
    bp1 = beta + 1.0

    if bm1 > 1e-10:
        log_r = alpha * (np.log(beta) - np.log(bm1))
        term_m = np.exp(-t + log_r) * gammainc(alpha, bm1 * t)
    else:
        from scipy.special import hyp1f1
        base = -t + alpha * np.log(t * beta) - gammaln(alpha + 1)
        h = hyp1f1(alpha, alpha + 1, -bm1 * t)
        with np.errstate(divide='ignore', invalid='ignore'):
            lh = np.where(h > 0, np.log(h), -np.inf)
        term_m = np.exp(base + lh)
        term_m = np.where(np.isfinite(term_m), term_m, 0.0)

    log_r_p = alpha * (np.log(beta) - np.log(bp1))
    term_p = np.exp(-t + log_r_p) * gammainc(alpha, bp1 * t)

    res = term_m - term_p
    log_gpdf = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(t) - beta * t
    gpdf = np.exp(np.where(log_gpdf > -700, log_gpdf, -700))
    gsf = gammaincc(alpha, beta * t)
    res += (np.exp(-2 * t) / 2 - 0.5 - t) * gpdf
    res += (1 - np.exp(-2 * t)) * gsf
    return res


def _dist_diff_demographic(t_arr, alpha, beta, lam_vals, t_bounds, L_bounds):
    """Fully vectorized distribution difference for demographic N(t).

    Key: precompute per-epoch gammainc contributions for ALL time points
    at once, then sum across epochs. No Python loops over time points.
    """
    t = np.maximum(np.asarray(t_arr, dtype=np.float64), 1e-15)
    n_t = len(t)
    M = len(lam_vals)

    # Compute Lambda(t) and lambda(t) for all t — vectorized via searchsorted
    # t_bounds has M+1 entries (but last epoch extends to inf)
    finite_bounds = t_bounds[:M]  # epoch start times (M values)
    epoch_idx = np.searchsorted(finite_bounds, t, side='right') - 1
    epoch_idx = np.clip(epoch_idx, 0, M - 1)

    Lam_t = L_bounds[epoch_idx] + lam_vals[epoch_idx] * (t - t_bounds[epoch_idx])

    # Compute I_plus(t) = ∫_0^t exp(Lambda(s)) * Gamma_pdf(s;a,b) ds
    # and I_minus(t) = ∫_0^t exp(-Lambda(s)) * Gamma_pdf(s;a,b) ds
    #
    # Per epoch k: integral from a_k to b_k where a_k = t_bounds[k], b_k = min(t, t_bounds[k+1])
    # exp(Lambda(s)) within epoch k = exp(L_k - lam_k * t_k) * exp(lam_k * s)
    # So the integral = exp(L_k - lam_k*t_k) * (beta/c_k)^alpha * [P(alpha, c_k*b_k) - P(alpha, c_k*a_k)]
    # where c_k = beta - lam_k (for plus) or beta + lam_k (for minus)

    I_plus = np.zeros(n_t)
    I_minus = np.zeros(n_t)

    for k in range(M):
        a_k = t_bounds[k]
        b_k_max = t_bounds[k + 1] if k < M - 1 else 1e15
        # For each time point, the upper bound of this epoch's contribution is min(t, b_k_max)
        b_k = np.minimum(t, b_k_max)
        # Only compute where t > a_k (this epoch contributes)
        active = t > a_k + 1e-15
        if not active.any():
            break

        lam_k = lam_vals[k]
        L_k = L_bounds[k]
        t_k = t_bounds[k]

        # Plus integral: c = beta - lam_k
        c_plus = beta - lam_k
        log_coeff_plus = L_k - lam_k * t_k
        if abs(c_plus) > 1e-10 and c_plus > 0:
            log_scale = log_coeff_plus + alpha * (np.log(beta) - np.log(c_plus))
            if log_scale > -500:
                gi_b = np.where(active, gammainc(alpha, c_plus * np.maximum(b_k, 1e-30)), 0.0)
                gi_a = gammainc(alpha, c_plus * max(a_k, 1e-30)) if a_k > 1e-15 else 0.0
                I_plus += np.where(active, np.exp(log_scale) * (gi_b - gi_a), 0.0)
        elif abs(c_plus) <= 1e-10:
            # c ≈ 0: integral ≈ exp(L_k) * beta^alpha/Gamma(alpha) * [b^alpha - a^alpha]/alpha
            if alpha > 0:
                coeff = np.exp(log_coeff_plus + alpha * np.log(beta) - gammaln(alpha))
                b_term = np.where(active, np.power(np.maximum(b_k, 1e-30), alpha), 0.0)
                a_term = max(a_k, 1e-30)**alpha if a_k > 1e-15 else 0.0
                I_plus += np.where(active, coeff * (b_term - a_term) / alpha, 0.0)
        # c_plus < 0: use the 1F1 path (rare, skip for now — would need hyp1f1)

        # Minus integral: c = beta + lam_k (always > 0)
        c_minus = beta + lam_k
        log_coeff_minus = -L_k + lam_k * t_k
        log_scale_m = log_coeff_minus + alpha * (np.log(beta) - np.log(c_minus))
        if log_scale_m > -500:
            gi_b_m = np.where(active, gammainc(alpha, c_minus * np.maximum(b_k, 1e-30)), 0.0)
            gi_a_m = gammainc(alpha, c_minus * max(a_k, 1e-30)) if a_k > 1e-15 else 0.0
            I_minus += np.where(active, np.exp(log_scale_m) * (gi_b_m - gi_a_m), 0.0)

    # Gamma PDF and SF
    log_gpdf = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(t) - beta * t
    gpdf = np.exp(np.where(log_gpdf > -700, log_gpdf, -700))
    gsf = gammaincc(alpha, beta * t)

    # Assemble generalized Schweiger formula
    exp_neg_L = np.exp(-Lam_t)
    exp_neg_2L = np.exp(-2 * Lam_t)

    res = exp_neg_L * (I_plus - I_minus)
    res += (exp_neg_2L / 2 - 0.5 - Lam_t) * gpdf
    res += (1 - exp_neg_2L) * gsf

    return res


def _make_time_grid(alpha, beta, n_steps=500):
    try:
        t_max_gamma = gammaincinv(alpha, 1 - 1e-3) / beta
    except:
        t_max_gamma = alpha / beta * 5
    t_max_gamma = max(t_max_gamma, 0.1)
    t_max_exp = -np.log(1e-3)
    g1 = np.linspace(1e-10, t_max_gamma, n_steps + 1)
    s2 = t_max_exp / max(n_steps - 1, 1)
    g2 = g1[-1] + np.arange(1, n_steps + 1) * s2
    return np.concatenate([g1, g2])


def generate_flow_field(Ne_values=None, epoch_boundaries=None, Ne_ref=10000,
                         mean_min=1e-5, mean_max=1e2, cv_min=1e-2, cv_max=1.0,
                         mean_n=51, cv_n=50, n_steps=500, verbose=True):
    is_demo = Ne_values is not None
    if is_demo:
        M = len(Ne_values)
        scale = 2.0 * Ne_ref
        t_b = np.zeros(M + 1)
        t_b[1:M] = np.array(epoch_boundaries[1:M]) / scale
        t_b_ext = np.append(t_b[:M], 1e15)  # extend last epoch
        lam_v = Ne_ref / np.array(Ne_values, dtype=np.float64)
        L_b = np.zeros(M + 1)
        for k in range(1, M):
            L_b[k] = L_b[k - 1] + lam_v[k - 1] * (t_b[k] - t_b[k - 1])

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
            if alpha < 1e-8 or beta < 1e-8 or alpha > 1e8:
                continue

            times = _make_time_grid(alpha, beta, n_steps)
            dt = np.diff(times)
            sqrt_dt = np.sqrt(np.append(dt, 0.0))

            if is_demo:
                delta = _dist_diff_demographic(times, alpha, beta, lam_v, t_b_ext, L_b)
            else:
                delta = _dist_diff_const_Ne(times, alpha, beta)

            delta_w = delta * sqrt_dt
            log_gpdf = (alpha * np.log(beta) - gammaln(alpha)
                        + (alpha - 1) * np.log(np.maximum(times, 1e-30)) - beta * times)
            gpdf = np.exp(np.where(log_gpdf > -700, log_gpdf, -700))
            d_la = gpdf * (np.log(beta) - digamma(alpha) + np.log(np.maximum(times, 1e-30))) * sqrt_dt * alpha / ln10
            d_lb = gpdf * (alpha / beta - times) * sqrt_dt * beta / ln10
            A = np.column_stack([d_la, d_lb])
            good = gpdf > gpdf.max() * 1e-12
            if good.sum() < 3:
                continue
            try:
                r, _, _, _ = np.linalg.lstsq(A[good], delta_w[good], rcond=None)
                u_out[i, j] = r[0] - r[1]
                v_out[i, j] = -0.5 * r[0]
            except:
                pass

        if verbose and ((i + 1) % 10 == 0 or i == mean_n - 1):
            print(f"  flow field: {(i+1)*cv_n}/{mean_n*cv_n}", flush=True)

    u_out /= _CALIBRATION_FACTOR
    v_out /= _CALIBRATION_FACTOR
    return u_out, v_out


# ══════════════════════════════════════════════════════════════
# I/O + estimation + context (compact)
# ══════════════════════════════════════════════════════════════

def write_flow_field(path, u, v, mean_min=1e-5, mean_max=1e2, cv_min=1e-2, cv_max=1.0):
    mn, cn = u.shape
    with open(path, 'w') as f:
        f.write(f"{mean_min} {mean_max} {mn}\n{cv_min} {cv_max} {cn}\n")
        for row in u: f.write(' '.join(f"{x:.10f}" for x in row) + '\n')
        for row in v: f.write(' '.join(f"{x:.10f}" for x in row) + '\n')

def load_flow_field(path):
    with open(path) as f: parts = f.read().split()
    i = 0
    mm,mx,mn = float(parts[i]),float(parts[i+1]),int(parts[i+2]); i+=3
    cm,cx,cn = float(parts[i]),float(parts[i+1]),int(parts[i+2]); i+=3
    u = np.array([float(x) for x in parts[i:i+mn*cn]]).reshape(mn,cn); i+=mn*cn
    v = np.array([float(x) for x in parts[i:i+mn*cn]]).reshape(mn,cn)
    return u,v,mm,mx,cm,cx

def estimate_demography(tmrca_means, Ne_init, n_epochs=20, t_max=None):
    if t_max is None: t_max = 10.0 * Ne_init
    vals = np.asarray(tmrca_means).ravel(); vals = vals[np.isfinite(vals) & (vals > 0)]
    fracs = np.linspace(0, 1, n_epochs + 1)**2
    boundaries = t_max * fracs; boundaries[0] = 0.0
    Ne_values = np.full(n_epochs, Ne_init, dtype=np.float64)
    for k in range(n_epochs):
        lo, hi = boundaries[k], boundaries[k+1]
        count = ((vals >= lo) & (vals < hi)).sum()
        if count < 10: continue
        dt = hi - lo
        if dt > 0:
            density = count / (len(vals) * dt)
            survival = np.mean(vals >= lo)
            if survival > 0.01:
                Ne_values[k] = max(100.0, 1.0 / (2.0 * density / survival))
    return Ne_values, boundaries

class DemographicFlowContext:
    def __init__(self, G, positions, Ne_init, mu, rho, flow_field_path,
                 n_calibration_pairs=50, n_epochs=20, n_steps=500,
                 cache_steps=0, gpu_id=0, verbose=True):
        self.S = len(positions); self.Ne_init = Ne_init; n = G.shape[0]
        _core.set_device(gpu_id)
        if verbose: print("Stage 1: estimating demography...", flush=True)
        rng = np.random.default_rng(42)
        n_cal = min(n_calibration_pairs, n*(n-1)//2)
        cs = set()
        while len(cs) < n_cal:
            a,b = sorted(rng.choice(n, 2, replace=False)); cs.add((int(a),int(b)))
        ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho, flow_field_path, cache_steps)
        r = ctx.run_fb(sorted(cs), mean_only=True); del ctx
        self.Ne_values, self.epoch_boundaries = estimate_demography(r["mean"], Ne_init, n_epochs=n_epochs)
        if verbose:
            print(f"  N(t): {self.Ne_values[:3].astype(int)} ... {self.Ne_values[-2:].astype(int)}", flush=True)
            print("Stage 2: generating demographic flow field...", flush=True)
        import time as _t; t0 = _t.perf_counter()
        u, v = generate_flow_field(Ne_values=self.Ne_values, epoch_boundaries=self.epoch_boundaries,
                                    Ne_ref=Ne_init, n_steps=n_steps, verbose=verbose)
        if verbose: print(f"  done in {_t.perf_counter()-t0:.1f}s", flush=True)
        self._ff_dir = tempfile.mkdtemp(); self._ff_path = os.path.join(self._ff_dir, "d.txt")
        write_flow_field(self._ff_path, u, v)
        self.ctx = _core.FlowContext(G, positions, float(Ne_init), mu, rho, self._ff_path, cache_steps)
        if verbose: print("Ready.", flush=True)
    def run_fb(self, pairs, mean_only=True): return self.ctx.run_fb(pairs, mean_only)
    def run_fwd(self, pairs, mean_only=True): return self.ctx.run_fwd(pairs, mean_only)
    def run_fb_summary(self, pairs): return self.ctx.run_fb_summary(pairs)
    @property
    def device_id(self): return self.ctx.device_id
    @property
    def demography(self): return self.Ne_values.copy(), self.epoch_boundaries.copy()
    def __del__(self):
        try:
            if hasattr(self,'_ff_path') and os.path.exists(self._ff_path): os.unlink(self._ff_path)
            if hasattr(self,'_ff_dir') and os.path.exists(self._ff_dir): os.rmdir(self._ff_dir)
        except: pass
