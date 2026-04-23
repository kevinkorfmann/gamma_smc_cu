"""
CoalescenceEstimator: high-level Python API for GPU-accelerated TMRCA inference.

Orchestrates the _core CUDA functions into a clean pipeline with three tiers:
  Tier 1 - Instant divergence (prefix scan / windowed divergence)
  Tier 2 - Changepoint segmentation (placeholder)
  Tier 3 - Full HMM forward-backward inference
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple

import gamma_smc_cu._core as _core


@dataclass
class TMRCAResult:
    """Result from Tier 3 HMM inference."""

    tmrca_mean: np.ndarray       # (n_pairs, S) posterior mean TMRCA
    tmrca_lower: np.ndarray      # (n_pairs, S) 2.5th percentile
    tmrca_upper: np.ndarray      # (n_pairs, S) 97.5th percentile
    pairs: list                  # list of (i, j) tuples
    positions: np.ndarray        # (S,) site positions in bp
    time_midpoints: np.ndarray   # (K,) time bin midpoints
    log_likelihood: np.ndarray   # (n_pairs,) per-pair log-likelihoods
    converged: bool
    n_iterations: int


@dataclass
class Segment:
    """A single TMRCA segment from Tier 2 changepoint segmentation."""

    start_bp: float
    end_bp: float
    start_idx: int
    end_idx: int
    tmrca: float
    n_mutations: int


def _compute_time_discretization(Ne: float, K: int, t_max: float):
    """
    Reproduce the quadratic time discretization used by the CUDA kernels.

    Returns
    -------
    boundaries : np.ndarray, shape (K+1,)
    midpoints  : np.ndarray, shape (K,)
    """
    fracs = np.arange(K + 1) / K
    boundaries = t_max * fracs ** 2
    midpoints = (boundaries[:-1] + boundaries[1:]) / 2.0
    return boundaries, midpoints


def _posterior_summaries(gamma: np.ndarray, midpoints: np.ndarray):
    """
    Compute posterior mean and 95% credible interval from gamma[S, K].

    Parameters
    ----------
    gamma : np.ndarray, shape (S, K)
        Posterior marginals at each site.
    midpoints : np.ndarray, shape (K,)
        Time bin midpoints.

    Returns
    -------
    mean  : np.ndarray, shape (S,)
    lower : np.ndarray, shape (S,)  — 2.5th percentile
    upper : np.ndarray, shape (S,)  — 97.5th percentile
    """
    S, K = gamma.shape

    # Posterior mean: sum_k gamma[s,k] * t_k
    mean = gamma @ midpoints  # (S,)

    # Credible intervals via cumulative posterior
    cum = np.cumsum(gamma, axis=1)  # (S, K)

    lower = np.empty(S)
    upper = np.empty(S)
    for s in range(S):
        idx_lo = np.searchsorted(cum[s], 0.025)
        idx_hi = np.searchsorted(cum[s], 0.975)
        lower[s] = midpoints[min(idx_lo, K - 1)]
        upper[s] = midpoints[min(idx_hi, K - 1)]

    return mean, lower, upper


def _default_pairs(n: int, max_pairs: int = 100) -> List[Tuple[int, int]]:
    """Generate a random sample of unique pairs from n haplotypes."""
    all_possible = n * (n - 1) // 2
    if all_possible <= max_pairs:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    rng = np.random.default_rng(seed=0)
    pair_set = set()
    while len(pair_set) < max_pairs:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        pair_set.add((int(i), int(j)))
    return sorted(pair_set)


class CoalescenceEstimator:
    """
    GPU-accelerated pairwise coalescence time estimator.

    Wraps the gamma_smc_cu._core CUDA functions into a pipeline supporting:
      - Tier 1: instant divergence statistics (site pi, windowed divergence)
      - Tier 3: full HMM forward-backward TMRCA inference
    """

    def __init__(
        self,
        genotypes: np.ndarray,
        positions: np.ndarray,
        mu: float = 1.25e-8,
        rho: float = 1e-8,
        Ne: float = 10_000,
        gpu_id: int = 0,
    ):
        """
        Parameters
        ----------
        genotypes : np.ndarray, shape (n, S), dtype uint8
            Haploid genotype matrix (0/1 entries).
        positions : np.ndarray, shape (S,), dtype float64
            Physical positions of segregating sites in base pairs.
        mu : float
            Per-site per-generation mutation rate.
        rho : float
            Per-site per-generation recombination rate.
        Ne : float
            Effective population size (constant).
        gpu_id : int
            GPU device to use (reserved for future multi-GPU support).
        """
        genotypes = np.ascontiguousarray(genotypes, dtype=np.uint8)
        positions = np.ascontiguousarray(positions, dtype=np.float64)

        if genotypes.ndim != 2:
            raise ValueError(f"genotypes must be 2D (n, S), got shape {genotypes.shape}")
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1D (S,), got shape {positions.shape}")
        if genotypes.shape[1] != positions.shape[0]:
            raise ValueError(
                f"genotypes has {genotypes.shape[1]} sites but positions has "
                f"{positions.shape[0]} entries"
            )

        self.genotypes = genotypes
        self.positions = positions
        self.n = genotypes.shape[0]
        self.S = genotypes.shape[1]
        self.mu = mu
        self.rho = rho
        self.Ne = Ne
        self.gpu_id = gpu_id

    @classmethod
    def from_tree_sequence(
        cls,
        ts,
        mu: Optional[float] = None,
        rho: Optional[float] = None,
        Ne: Optional[float] = None,
        gpu_id: int = 0,
    ) -> "CoalescenceEstimator":
        """
        Construct a CoalescenceEstimator from a tskit TreeSequence.

        Parameters
        ----------
        ts : tskit.TreeSequence
            Tree sequence with mutations.
        mu : float, optional
            Mutation rate. If None, estimated from ts metadata or defaults to
            1.25e-8.
        rho : float, optional
            Recombination rate. If None, estimated from ts metadata or defaults
            to 1e-8.
        Ne : float, optional
            Effective population size. If None, estimated from tree sequence
            diversity or defaults to 10000.
        gpu_id : int
            GPU device id.

        Returns
        -------
        CoalescenceEstimator
        """
        # Extract genotype matrix: tskit returns (n_sites, n_samples), we need
        # (n_samples, n_sites).
        G = ts.genotype_matrix().T.astype(np.uint8)
        positions = np.array([v.position for v in ts.variants()], dtype=np.float64)

        # Infer parameters from tree sequence if not provided
        if mu is None:
            # Try to get from provenance / metadata; fall back to a reasonable
            # default for human data.
            mu = 1.25e-8

        if rho is None:
            rho = 1e-8

        if Ne is None:
            # Rough estimate: Watterson's theta -> Ne
            # theta_W = S / a_n, Ne ~ theta_W / (4 * mu * L)
            n_haps = G.shape[0]
            n_sites = G.shape[1]
            seq_len = ts.sequence_length
            a_n = sum(1.0 / k for k in range(1, n_haps))
            theta_w = n_sites / a_n
            Ne_est = theta_w / (4.0 * mu * seq_len)
            Ne = max(Ne_est, 100.0)  # floor at 100

        return cls(G, positions, mu=mu, rho=rho, Ne=Ne, gpu_id=gpu_id)

    # ------------------------------------------------------------------
    # Tier 1: instant divergence
    # ------------------------------------------------------------------

    def site_pi(self) -> np.ndarray:
        """
        Tier 1: Per-site nucleotide diversity pi(s).

        Computes the average pairwise divergence across a sample of pairs at
        each site.

        Returns
        -------
        pi : np.ndarray, shape (S,)
            Estimated nucleotide diversity at each segregating site.
        """
        # Use the SFS to compute pi analytically:
        # pi = sum_k 2*k*(n-k) / (n*(n-1)) / S
        # But per-site pi for biallelic sites is just:
        # pi(s) = 2 * freq(s) * (1 - freq(s)) * n / (n - 1)
        freq = self.genotypes.mean(axis=0).astype(np.float64)
        pi = 2.0 * freq * (1.0 - freq) * self.n / (self.n - 1)
        return pi

    def pairwise_divergence(
        self,
        pairs: Optional[List[Tuple[int, int]]] = None,
        window_sizes: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Tier 1: Windowed pairwise divergence via GPU prefix scan.

        Parameters
        ----------
        pairs : list of (int, int), optional
            Pairs of haplotype indices. If None, a default random sample of up
            to 100 pairs is used.
        window_sizes : list of int, optional
            Window sizes in number of sites. If None, defaults to [100].
            Multiple window sizes are stacked along a third axis.

        Returns
        -------
        divergence : np.ndarray
            If one window size: shape (n_pairs, S).
            If multiple: shape (n_pairs, S, n_windows).
        """
        if pairs is None:
            pairs = _default_pairs(self.n)
        if window_sizes is None:
            window_sizes = [100]

        results = []
        for ws in window_sizes:
            div = np.array(
                _core.windowed_divergence(self.genotypes, pairs, ws)
            )
            # Normalize counts to fraction of differing sites
            div = div / float(ws)
            results.append(div)

        if len(results) == 1:
            return results[0]
        else:
            return np.stack(results, axis=-1)

    # ------------------------------------------------------------------
    # Tier 3: HMM forward-backward inference
    # ------------------------------------------------------------------

    def infer_tmrca(
        self,
        pairs: Optional[List[Tuple[int, int]]] = None,
        max_iterations: int = 5,
        n_time_bins: int = 32,
        damping: float = 0.5,
        convergence_tol: float = 0.01,
        t_max: Optional[float] = None,
    ) -> TMRCAResult:
        """
        Tier 3: Full HMM forward-backward TMRCA inference.

        Runs the SMC HMM for each pair independently and extracts posterior
        summaries.  Future versions will incorporate ultrametric EP messages
        across pairs and iterative Ne estimation.

        Parameters
        ----------
        pairs : list of (int, int), optional
            Pairs of haplotype indices. If None, uses a default random sample.
        max_iterations : int
            Maximum EP iterations (currently only 1 pass is used).
        n_time_bins : int
            Number of discrete time bins K. Must be 32 (compiled constant).
        damping : float
            EP damping factor (reserved for future EP integration).
        convergence_tol : float
            Convergence tolerance on log-likelihood change (reserved).
        t_max : float, optional
            Maximum coalescence time. Defaults to 10 * Ne.

        Returns
        -------
        TMRCAResult
            Posterior summaries for all requested pairs.
        """
        if pairs is None:
            pairs = _default_pairs(self.n)
        if t_max is None:
            t_max = 10.0 * self.Ne

        _, midpoints = _compute_time_discretization(self.Ne, n_time_bins, t_max)

        n_pairs = len(pairs)
        tmrca_mean = np.empty((n_pairs, self.S))
        tmrca_lower = np.empty((n_pairs, self.S))
        tmrca_upper = np.empty((n_pairs, self.S))
        log_liks = np.empty(n_pairs)

        prev_total_ll = -np.inf
        converged = False
        n_iter = 0

        for iteration in range(max_iterations):
            n_iter = iteration + 1

            for p_idx, pair in enumerate(pairs):
                gamma = np.array(
                    _core.hmm_posterior(
                        self.genotypes,
                        self.positions,
                        pair,
                        K=n_time_bins,
                        Ne=self.Ne,
                        mu=self.mu,
                        rho=self.rho,
                        t_max=t_max,
                    )
                )  # (S, K)

                mean, lower, upper = _posterior_summaries(gamma, midpoints)
                tmrca_mean[p_idx] = mean
                tmrca_lower[p_idx] = lower
                tmrca_upper[p_idx] = upper

                log_liks[p_idx] = _core.hmm_log_likelihood(
                    self.genotypes,
                    self.positions,
                    pair,
                    K=n_time_bins,
                    Ne=self.Ne,
                    mu=self.mu,
                    rho=self.rho,
                    t_max=t_max,
                )

            total_ll = log_liks.sum()

            # Check convergence (relative change in total log-likelihood)
            if iteration > 0:
                rel_change = abs(total_ll - prev_total_ll) / (
                    abs(prev_total_ll) + 1e-30
                )
                if rel_change < convergence_tol:
                    converged = True
                    break

            prev_total_ll = total_ll

            # Future: update Ne estimate, integrate EP messages, apply damping.
            # For now, single-pass HMM — break after first iteration.
            break

        return TMRCAResult(
            tmrca_mean=tmrca_mean,
            tmrca_lower=tmrca_lower,
            tmrca_upper=tmrca_upper,
            pairs=list(pairs),
            positions=self.positions.copy(),
            time_midpoints=midpoints,
            log_likelihood=log_liks,
            converged=converged,
            n_iterations=n_iter,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def tmrca_landscape(
        self, n_pairs: int = 100, window_bp: float = 10_000
    ) -> np.ndarray:
        """
        Genome-wide TMRCA landscape averaged over random pairs.

        Uses Tier 1 windowed divergence scaled by 1/(2*mu) as a fast TMRCA
        proxy.

        Parameters
        ----------
        n_pairs : int
            Number of random pairs to average over.
        window_bp : float
            Window size in base pairs for smoothing.

        Returns
        -------
        landscape : np.ndarray, shape (S,)
            Smoothed average TMRCA estimate at each site.
        """
        pairs = _default_pairs(self.n, max_pairs=n_pairs)

        # Convert window_bp to window in number of sites
        if self.S < 2:
            window_sites = 1
        else:
            mean_spacing = (self.positions[-1] - self.positions[0]) / (self.S - 1)
            window_sites = max(1, int(round(window_bp / mean_spacing)))

        div = np.array(
            _core.windowed_divergence(self.genotypes, pairs, window_sites)
        )  # (n_pairs, S) — raw XOR mismatch counts in window

        # Convert to per-bp divergence: count / window_bp
        # Each window spans window_sites sites; approximate window bp from
        # mean inter-site spacing.
        if self.S >= 2:
            mean_spacing = (self.positions[-1] - self.positions[0]) / (self.S - 1)
        else:
            mean_spacing = 1.0
        window_bp_actual = window_sites * mean_spacing

        # Scale divergence to TMRCA: E[d_bp] ~ 2*mu*T => T ~ d_bp / (2*mu)
        avg_counts = div.mean(axis=0)
        if self.mu > 0 and window_bp_actual > 0:
            landscape = avg_counts / (window_bp_actual * 2.0 * self.mu)
        else:
            landscape = avg_counts

        return landscape

    def sfs(self) -> np.ndarray:
        """
        Compute the site frequency spectrum on GPU.

        Returns
        -------
        sfs : np.ndarray, shape (n+1,)
            Count of sites with k derived alleles, for k = 0 .. n.
        """
        return np.array(_core.compute_sfs(self.genotypes))
