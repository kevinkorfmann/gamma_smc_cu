"""
Pure-numpy reference implementation of Gamma-SMC forward(-backward) filtering.

Maintains Gamma(alpha, beta) posterior over TMRCA at each site.
Conjugate to Poisson emission model with moment-matched recombination transitions.
"""

import numpy as np


def _moment_match(alpha, beta, p, inv_lambda, prior_m2):
    """Moment-match mixture (1-p)*Gamma(alpha,beta) + p*Gamma(1,lambda)."""
    if p < 1e-7:
        return alpha, beta
    ib = 1.0 / beta
    filt_mean = alpha * ib
    m1 = (1.0 - p) * filt_mean + p * inv_lambda
    m2 = ((1.0 - p) * alpha * (alpha + 1.0) * ib * ib
          + p * prior_m2)
    var = m2 - m1 * m1
    if var > 1e-30:
        beta_new = m1 / var
        alpha_new = m1 * beta_new
        return alpha_new, beta_new
    return alpha, beta


def gamma_smc_forward(G, positions, pair, Ne=10_000, mu=1.25e-8, rho=1e-8):
    """
    Gamma-SMC forward filtering for a single pair.

    Parameters
    ----------
    G : ndarray (n, S), uint8
        Genotype matrix (0/1).
    positions : ndarray (S,), float64
        Physical positions of segregating sites.
    pair : tuple (i, j)
        Haplotype indices.
    Ne : float
        Effective population size.
    mu : float
        Per-base mutation rate.
    rho : float
        Per-base recombination rate.

    Returns
    -------
    mean : ndarray (S,)
    lower : ndarray (S,)
    upper : ndarray (S,)
    """
    hi, hj = pair
    S = len(positions)
    xor = (G[hi] ^ G[hj]).astype(np.int32)

    lam = 1.0 / (2.0 * Ne)
    two_mu = 2.0 * mu
    inv_lambda = 2.0 * Ne

    alpha = 1.0
    beta = lam

    means = np.empty(S, dtype=np.float64)
    lowers = np.empty(S, dtype=np.float64)
    uppers = np.empty(S, dtype=np.float64)

    prev_pos = 0.0

    for s in range(S):
        pos = positions[s]
        gap = pos - prev_pos
        prev_pos = pos

        if s > 0:
            # 1. Recombination transition (moment-match)
            p = 1.0 - np.exp(-rho * gap)
            if p > 1e-7:
                ib = 1.0 / beta
                filt_mean = alpha * ib
                prior_mean = inv_lambda

                m1 = (1.0 - p) * filt_mean + p * prior_mean
                m2 = ((1.0 - p) * alpha * (alpha + 1.0) * ib * ib
                      + p * 2.0 * inv_lambda * inv_lambda)

                var = m2 - m1 * m1
                if var > 1e-30:
                    beta = m1 / var
                    alpha = m1 * beta

            # 2. Gap emission
            beta += two_mu * gap

        # 3. Site emission
        if xor[s]:
            alpha += 1.0

        # 4. Output
        mean = alpha / beta
        means[s] = mean

        # Wilson-Hilferty CI
        inv9a = 1.0 / (9.0 * alpha)
        sq = np.sqrt(inv9a)
        base = 1.0 - inv9a
        lo_factor = max(base - 1.96 * sq, 0.0)
        hi_factor = base + 1.96 * sq
        lowers[s] = max(mean * lo_factor**3, 0.0)
        uppers[s] = mean * hi_factor**3

    return means, lowers, uppers


def gamma_smc_forward_backward(G, positions, pair, Ne=10_000, mu=1.25e-8, rho=1e-8):
    """
    Gamma-SMC forward-backward smoothing for a single pair.

    Two-pass algorithm:
    1. Forward pass (left→right): store (alpha_f, beta_f) at each site
    2. Backward pass (right→left): mirror of forward, store (alpha_b, beta_b)
    3. Combine: Gamma(alpha_f + alpha_b - 1, beta_f + beta_b - lambda)

    The backward pass is a "backward filter" — same recursion run in reverse.
    Both filters include the prior. The combination removes one copy of the prior.
    The backward filter does NOT include the emission at site s; it is stored
    before absorbing site s's emission.

    Returns
    -------
    mean : ndarray (S,)
    lower : ndarray (S,)
    upper : ndarray (S,)
    """
    hi, hj = pair
    S = len(positions)
    xor = (G[hi] ^ G[hj]).astype(np.int32)

    lam = 1.0 / (2.0 * Ne)
    two_mu = 2.0 * mu
    inv_lambda = 2.0 * Ne
    prior_m2 = 2.0 * inv_lambda * inv_lambda

    # --- Forward pass ---
    alpha_f = np.empty(S, dtype=np.float64)
    beta_f = np.empty(S, dtype=np.float64)

    alpha = 1.0
    beta = lam
    prev_pos = 0.0

    for s in range(S):
        pos = positions[s]
        gap = pos - prev_pos
        prev_pos = pos

        if s > 0:
            p = 1.0 - np.exp(-rho * gap)
            alpha, beta = _moment_match(alpha, beta, p, inv_lambda, prior_m2)
            beta += two_mu * gap

        alpha += xor[s]
        alpha_f[s] = alpha
        beta_f[s] = beta

    # --- Backward pass (right to left) ---
    alpha_b = np.empty(S, dtype=np.float64)
    beta_b = np.empty(S, dtype=np.float64)

    alpha = 1.0
    beta = lam

    for s in range(S - 1, -1, -1):
        # Store BEFORE absorbing emission at site s
        alpha_b[s] = alpha
        beta_b[s] = beta

        # Absorb emission at s, then transition to s-1
        alpha += xor[s]

        if s > 0:
            gap = positions[s] - positions[s - 1]
            beta += two_mu * gap
            p = 1.0 - np.exp(-rho * gap)
            alpha, beta = _moment_match(alpha, beta, p, inv_lambda, prior_m2)

    # --- Combine ---
    alpha_smooth = alpha_f + alpha_b - 1.0
    beta_smooth = beta_f + beta_b - lam

    # Ensure valid Gamma parameters
    alpha_smooth = np.maximum(alpha_smooth, 1.0)
    beta_smooth = np.maximum(beta_smooth, 1e-30)

    means = alpha_smooth / beta_smooth

    # Wilson-Hilferty CI
    inv9a = 1.0 / (9.0 * alpha_smooth)
    sq = np.sqrt(inv9a)
    base = 1.0 - inv9a
    lo_factor = np.maximum(base - 1.96 * sq, 0.0)
    hi_factor = base + 1.96 * sq
    lowers = np.maximum(means * lo_factor ** 3, 0.0)
    uppers = means * hi_factor ** 3

    return means, lowers, uppers
