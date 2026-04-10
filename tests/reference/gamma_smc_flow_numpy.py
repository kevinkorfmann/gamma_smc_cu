"""
Pure-numpy reference implementation of flow-field Gamma-SMC forward-backward.

Uses the same base flow field as Schweiger's implementation.
Operates in (log10_mean, log10_cv) coordinate space with bilinear interpolation.
"""

import numpy as np


def load_flow_field(path):
    """Load base flow field from Schweiger's text file."""
    with open(path) as f:
        lines = f.read().split('\n')

    # Line 1: mean grid
    parts = lines[0].split()
    mean_min, mean_max, mean_n = float(parts[0]), float(parts[1]), int(parts[2])

    # Line 2: cv grid
    parts = lines[1].split()
    cv_min, cv_max, cv_n = float(parts[0]), float(parts[1]), int(parts[2])

    mean_log10_min = np.log10(mean_min)
    mean_log10_max = np.log10(mean_max)
    cv_log10_min = np.log10(cv_min)
    cv_log10_max = np.log10(cv_max)

    # U values: lines 2..2+mean_n
    u = np.zeros((mean_n, cv_n))
    for r in range(mean_n):
        vals = lines[2 + r].split()
        u[r] = [float(v) for v in vals]

    # V values: lines 2+mean_n..
    v = np.zeros((mean_n, cv_n))
    for r in range(mean_n):
        vals = lines[2 + mean_n + r].split()
        v[r] = [float(vv) for vv in vals]

    return {
        'u': u, 'v': v,
        'mean_n': mean_n, 'cv_n': cv_n,
        'mean_log10_min': mean_log10_min,
        'mean_log10_max': mean_log10_max,
        'cv_log10_min': cv_log10_min,
        'cv_log10_max': cv_log10_max,
    }


def _bilinear(table, mean_log10, cv_log10, ff):
    """Bilinear interpolation on flow field grid."""
    mn, cn = ff['mean_n'], ff['cv_n']
    m_step = (ff['mean_log10_max'] - ff['mean_log10_min']) / (mn - 1)
    c_step = (ff['cv_log10_max'] - ff['cv_log10_min']) / (cn - 1)

    fm = (mean_log10 - ff['mean_log10_min']) / m_step
    fc = (cv_log10 - ff['cv_log10_min']) / c_step

    fm = np.clip(fm, 0, mn - 1)
    fc = np.clip(fc, 0, cn - 1)

    m0 = int(fm)
    c0 = int(fc)
    if m0 == mn - 1:
        m0 -= 1
    if c0 == cn - 1:
        c0 -= 1
    m1 = min(m0 + 1, mn - 1)
    c1 = min(c0 + 1, cn - 1)
    wm = fm - m0
    wc = fc - c0

    v00 = table[m0, c0]
    v01 = table[m0, c1]
    v10 = table[m1, c0]
    v11 = table[m1, c1]

    return (1 - wm) * ((1 - wc) * v00 + wc * v01) + wm * ((1 - wc) * v10 + wc * v11)


def _flow_field_advance(mean_log10, cv_log10, ff, scaled_rho_total, scaled_mu_total,
                        max_step=0.1):
    """Apply flow field for a gap with adaptive sub-stepping."""
    if scaled_rho_total < 1e-12:
        return mean_log10, cv_log10

    n_iter = max(1, int(np.ceil(scaled_rho_total / max_step)))
    rho_per = scaled_rho_total / n_iter
    mu_per = scaled_mu_total / n_iter

    for _ in range(n_iter):
        # Mutation emission
        a_log = -2.0 * cv_log10
        b_log = a_log - mean_log10
        b_lin = 10.0**b_log + mu_per
        b_log = np.log10(b_lin)
        mean_log10 = a_log - b_log

        # Recombination via flow field
        u = _bilinear(ff['u'], mean_log10, cv_log10, ff)
        v = _bilinear(ff['v'], mean_log10, cv_log10, ff)
        mean_log10 += u * rho_per
        cv_log10 += v * rho_per

        mean_log10 = np.clip(mean_log10, ff['mean_log10_min'], ff['mean_log10_max'])
        cv_log10 = np.clip(cv_log10, ff['cv_log10_min'], ff['cv_log10_max'])

    return mean_log10, cv_log10


def gamma_smc_flow_fb(G, positions, pair, Ne=10_000, mu=1.25e-8, rho=1e-8,
                      flow_field_path='/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt'):
    """
    Flow-field Gamma-SMC forward-backward for a single pair.

    Returns means, lowers, uppers (in generations).
    """
    ff = load_flow_field(flow_field_path)
    hi, hj = pair
    S = len(positions)
    xor = (G[hi] ^ G[hj]).astype(np.int32)

    lam = 1.0 / (2.0 * Ne)
    scaled_rho_per_bp = 4.0 * Ne * rho
    scaled_mu_per_bp = 2.0 * mu / lam  # = 4*Ne*mu

    # Forward pass
    fwd_m = np.empty(S)
    fwd_c = np.empty(S)

    mean_log10 = 0.0
    cv_log10 = 0.0
    prev_pos = 0.0

    for s in range(S):
        pos = positions[s]
        gap = pos - prev_pos
        prev_pos = pos

        if s > 0 and gap > 0:
            mean_log10, cv_log10 = _flow_field_advance(
                mean_log10, cv_log10, ff,
                scaled_rho_per_bp * gap,
                scaled_mu_per_bp * gap)

        # Site emission: upstream gamma_smc applies beta += mu at every
        # observed site, and het sites additionally apply alpha += 1.
        a_log = -2.0 * cv_log10
        beta_log10 = a_log - mean_log10
        beta_log10 = np.log10(10.0**beta_log10 + scaled_mu_per_bp)
        if xor[s]:
            a_log = np.log10(10.0**a_log + 1.0)
        mean_log10 = a_log - beta_log10
        cv_log10 = -0.5 * a_log

        fwd_m[s] = mean_log10
        fwd_c[s] = cv_log10

    # Backward pass
    mean_log10 = 0.0
    cv_log10 = 0.0

    means = np.empty(S)
    lowers = np.empty(S)
    uppers = np.empty(S)

    for s in range(S - 1, -1, -1):
        # Backward state before emission
        bwd_a = 10.0**(-2.0 * cv_log10)
        bwd_b = 10.0**(-2.0 * cv_log10 - mean_log10)

        # Forward state
        fwd_a = 10.0**(-2.0 * fwd_c[s])
        fwd_b = 10.0**(-2.0 * fwd_c[s] - fwd_m[s])

        # Combine
        a_s = max(fwd_a + bwd_a - 1.0, 1.0)
        b_s = max(fwd_b + bwd_b - 1.0, 1e-10)
        mean_gen = (a_s / b_s) * 2.0 * Ne
        means[s] = mean_gen

        # Wilson-Hilferty CI
        inv9a = 1.0 / (9.0 * a_s)
        sq = np.sqrt(inv9a)
        base = 1.0 - inv9a
        lo_f = max(base - 1.96 * sq, 0.0)
        hi_f = base + 1.96 * sq
        lowers[s] = max(mean_gen * lo_f**3, 0.0)
        uppers[s] = mean_gen * hi_f**3

        # Absorb emission with the same site semantics as the forward pass.
        a_log = -2.0 * cv_log10
        beta_log10 = a_log - mean_log10
        beta_log10 = np.log10(10.0**beta_log10 + scaled_mu_per_bp)
        if xor[s]:
            a_log = np.log10(10.0**a_log + 1.0)
        mean_log10 = a_log - beta_log10
        cv_log10 = -0.5 * a_log

        # Transition
        if s > 0:
            gap = positions[s] - positions[s - 1]
            if gap > 0:
                mean_log10, cv_log10 = _flow_field_advance(
                    mean_log10, cv_log10, ff,
                    scaled_rho_per_bp * gap,
                    scaled_mu_per_bp * gap)

    return means, lowers, uppers
