"""
Pure-numpy reference implementation of the SMC HMM forward-backward algorithm.
Used for validating the CUDA kernels.
"""

import numpy as np


class NumpyHMM:
    """Reference HMM for a single pair under the Sequentially Markov Coalescent."""

    def __init__(self, G, positions, mu=1.25e-8, rho=1e-8, K=32, Ne=10_000, t_max=None):
        """
        Parameters
        ----------
        G : np.ndarray, shape (n, S), dtype uint8
            Genotype matrix (haploid).
        positions : np.ndarray, shape (S,), dtype float64
            Physical positions of segregating sites.
        mu : float
            Per-site per-generation mutation rate.
        rho : float
            Per-site per-generation recombination rate.
        K : int
            Number of discrete time bins.
        Ne : float
            Effective population size (constant).
        t_max : float or None
            Maximum coalescence time. Defaults to 10 * Ne.
        """
        self.G = G
        self.positions = positions
        self.mu = mu
        self.rho = rho
        self.K = K
        self.Ne = Ne
        self.t_max = t_max if t_max is not None else 10.0 * Ne
        self.S = G.shape[1]

        # Time discretization (quadratic spacing)
        self.time_boundaries = np.array(
            [self.t_max * (k / K) ** 2 for k in range(K + 1)]
        )
        self.time_midpoints = (self.time_boundaries[:-1] + self.time_boundaries[1:]) / 2.0

        # Coalescent prior
        self.coal_prior = self._compute_prior()

        # Cumulative recombination distances
        self.cum_rho = np.zeros(self.S)
        for s in range(1, self.S):
            self.cum_rho[s] = self.cum_rho[s - 1] + rho * (positions[s] - positions[s - 1])

    def _compute_prior(self):
        """Coalescent prior q[k] under constant Ne."""
        q = np.zeros(self.K)
        for k in range(self.K):
            t_lo = self.time_boundaries[k]
            t_hi = self.time_boundaries[k + 1]
            q[k] = np.exp(-t_lo / (2.0 * self.Ne)) - np.exp(-t_hi / (2.0 * self.Ne))
        q /= q.sum()
        return q

    def _emission(self, d, s, k):
        """Emission probability P(d_s | T = t_k)."""
        t_k = self.time_midpoints[k]
        if d == 1:
            return 1.0 - np.exp(-2.0 * self.mu * t_k)
        else:
            return np.exp(-2.0 * self.mu * t_k)

    def _gap_emission(self, s, k):
        """Gap emission: P(no mutations between sites s-1 and s | T = t_k)."""
        if s == 0:
            return 1.0
        gap_bp = self.positions[s] - self.positions[s - 1] - 1.0
        if gap_bp <= 0:
            return 1.0
        t_k = self.time_midpoints[k]
        mu_gap = self.mu  # uniform rate
        return np.exp(-2.0 * mu_gap * t_k * gap_bp)

    def forward(self, pair):
        """
        Forward algorithm.

        Returns
        -------
        alpha : np.ndarray, shape (S, K)
            Normalized forward probabilities.
        """
        i, j = pair
        xor = np.bitwise_xor(self.G[i], self.G[j]).astype(int)

        alpha = np.zeros((self.S, self.K))

        # Site 0: prior * emission
        for k in range(self.K):
            alpha[0, k] = self.coal_prior[k] * self._emission(xor[0], 0, k)
        s0_sum = alpha[0].sum()
        if s0_sum > 0:
            alpha[0] /= s0_sum

        for s in range(1, self.S):
            r = self.cum_rho[s] - self.cum_rho[s - 1]

            for k in range(self.K):
                t_k = self.time_midpoints[k]
                no_recomb = np.exp(-r * t_k)

                # Transition
                stay = no_recomb * alpha[s - 1, k]
                recomb_mass = 0.0
                for l in range(self.K):
                    t_l = self.time_midpoints[l]
                    recomb_mass += (1.0 - np.exp(-r * t_l)) * alpha[s - 1, l]
                alpha[s, k] = stay + self.coal_prior[k] * recomb_mass

                # Gap emission
                alpha[s, k] *= self._gap_emission(s, k)

                # Site emission
                alpha[s, k] *= self._emission(xor[s], s, k)

            s_sum = alpha[s].sum()
            if s_sum > 0:
                alpha[s] /= s_sum

        return alpha

    def backward(self, pair):
        """
        Backward algorithm.

        Returns
        -------
        beta : np.ndarray, shape (S, K)
            Normalized backward probabilities.
        """
        i, j = pair
        xor = np.bitwise_xor(self.G[i], self.G[j]).astype(int)

        beta = np.zeros((self.S, self.K))
        beta[self.S - 1, :] = 1.0

        for s in range(self.S - 2, -1, -1):
            r = self.cum_rho[s + 1] - self.cum_rho[s]

            for k in range(self.K):
                t_k = self.time_midpoints[k]
                no_recomb = np.exp(-r * t_k)

                # beta_new[k] = sum_l A[k][l] * emit(l, s+1) * gap_emit(l, s+1) * beta[l]
                val = 0.0
                for l in range(self.K):
                    t_l = self.time_midpoints[l]
                    emit_l = self._emission(xor[s + 1], s + 1, l)
                    gap_l = self._gap_emission(s + 1, l)

                    if k == l:
                        trans = np.exp(-r * t_k) + (1.0 - np.exp(-r * t_k)) * self.coal_prior[l]
                    else:
                        trans = (1.0 - np.exp(-r * t_k)) * self.coal_prior[l]

                    val += trans * emit_l * gap_l * beta[s + 1, l]

                beta[s, k] = val

            s_sum = beta[s].sum()
            if s_sum > 0:
                beta[s] /= s_sum

        return beta

    def posterior(self, pair):
        """
        Posterior marginals gamma[s, k] = P(T(s) = t_k | data).

        Returns
        -------
        gamma : np.ndarray, shape (S, K)
        """
        alpha = self.forward(pair)
        beta = self.backward(pair)

        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        gamma /= row_sums
        return gamma

    def log_likelihood(self, pair):
        """Total log-likelihood of the data for this pair."""
        i, j = pair
        xor = np.bitwise_xor(self.G[i], self.G[j]).astype(int)

        alpha = np.zeros(self.K)
        for k in range(self.K):
            alpha[k] = self.coal_prior[k] * self._emission(xor[0], 0, k)
        s0_sum = alpha.sum()
        if s0_sum > 0:
            alpha /= s0_sum
        log_lik = np.log(s0_sum) if s0_sum > 0 else -1e30

        for s in range(1, self.S):
            r = self.cum_rho[s] - self.cum_rho[s - 1]
            alpha_new = np.zeros(self.K)

            for k in range(self.K):
                t_k = self.time_midpoints[k]
                no_recomb = np.exp(-r * t_k)
                stay = no_recomb * alpha[k]
                recomb_mass = 0.0
                for l in range(self.K):
                    t_l = self.time_midpoints[l]
                    recomb_mass += (1.0 - np.exp(-r * t_l)) * alpha[l]
                alpha_new[k] = stay + self.coal_prior[k] * recomb_mass
                alpha_new[k] *= self._gap_emission(s, k)
                alpha_new[k] *= self._emission(xor[s], s, k)

            s_sum = alpha_new.sum()
            if s_sum > 0:
                alpha_new /= s_sum
            log_lik += np.log(s_sum) if s_sum > 0 else -1e30
            alpha = alpha_new

        return log_lik
