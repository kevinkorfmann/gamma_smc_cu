"""
Pure-numpy reference implementation of the PELT (Pruned Exact Linear Time)
changepoint detection algorithm for pairwise TMRCA estimation.

Used for validating the CUDA tier-2 PELT kernel.
"""

import numpy as np


class NumpyPELT:
    """
    PELT changepoint detection on a binary XOR signal under a Poisson
    mutation model.

    For each segment [a, b) with mutation count C and physical length L:
      - MLE rate: lambda_hat = C / L
      - Segment cost (negative Poisson log-likelihood):
            C == 0  =>  2 * mu * L      (expected count under null)
            C >  0  =>  -C * log(C/L) + C
      - MLE TMRCA: T = C / (2 * mu * L)

    Parameters
    ----------
    mu : float
        Per-bp per-generation mutation rate.
    """

    def __init__(self, mu=1.25e-8):
        self.mu = mu

    def segment_cost(self, count, length_bp):
        """
        Poisson negative log-likelihood for a single segment.

        Parameters
        ----------
        count : int
            Number of mutations (XOR differences) in the segment.
        length_bp : float
            Physical length of the segment in base pairs.

        Returns
        -------
        float
            Segment cost.
        """
        if length_bp <= 0.0:
            return 1e30
        if count == 0:
            return 2.0 * self.mu * length_bp
        rate = count / length_bp
        return -count * np.log(rate) + count

    def detect(self, xor_signal, positions, penalty=None):
        """
        PELT changepoint detection.

        Parameters
        ----------
        xor_signal : np.ndarray, shape (S,), dtype uint8 or int
            Binary XOR signal (1 = difference, 0 = same) at each
            segregating site.
        positions : np.ndarray, shape (S,), dtype float64
            Physical positions of each segregating site (in bp).
        penalty : float or None
            BIC penalty per changepoint. Default: log(S).

        Returns
        -------
        segments : list of (start_idx, end_idx, tmrca_mle, n_mutations)
            Each segment is a half-open interval [start_idx, end_idx).
            tmrca_mle is the MLE TMRCA in generations, and n_mutations
            is the number of XOR differences in the segment.
        """
        S = len(xor_signal)
        if S == 0:
            return []

        if penalty is None:
            penalty = np.log(S)

        # Prefix sum of XOR differences
        prefix = np.zeros(S + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(xor_signal)

        # F[s] = optimal cost for sites 0..s-1 (using 1-indexed: data points 1..s)
        # We work with 0-indexed sites: F[0] = 0 (empty), F[s] = best cost for [0, s)
        # Actually align with the CUDA kernel: F[s] = best cost ending at site s.
        F = np.full(S, np.inf)
        last_cp = np.zeros(S, dtype=int)

        # Pruning set R: candidate changepoint starts
        R = [0]

        # F[0] = 0 (base case: cost of reaching site 0 is 0)
        F[0] = 0.0
        last_cp[0] = 0

        for s in range(1, S):
            best_cost = np.inf
            best_t = 0

            for t in R:
                count = int(prefix[s] - prefix[t])
                length = positions[s] - positions[t]
                c = self.segment_cost(count, length)
                total = F[t] + c + penalty
                if total < best_cost:
                    best_cost = total
                    best_t = t

            F[s] = best_cost
            last_cp[s] = best_t

            # Prune: keep t where F[t] + cost(t, s) <= F[s]
            R_new = []
            for t in R:
                count = int(prefix[s] - prefix[t])
                length = positions[s] - positions[t]
                c = self.segment_cost(count, length)
                if F[t] + c <= F[s]:
                    R_new.append(t)
            R_new.append(s)
            R = R_new

        # Traceback
        changepoints = []
        pos = S - 1
        while pos > 0:
            changepoints.append(pos)
            pos = last_cp[pos]
        changepoints.append(0)
        changepoints.reverse()

        # Build segments
        segments = []
        for i in range(len(changepoints) - 1):
            seg_start = changepoints[i]
            seg_end = changepoints[i + 1]
            count = int(prefix[seg_end] - prefix[seg_start])
            length = positions[seg_end] - positions[seg_start]
            if count == 0 or length <= 0.0:
                tmrca_mle = 0.0
            else:
                tmrca_mle = count / (2.0 * self.mu * length)
            segments.append((seg_start, seg_end, tmrca_mle, count))

        return segments

    def detect_from_prefix(self, prefix, positions, penalty=None):
        """
        PELT detection from a pre-computed prefix sum array.

        Parameters
        ----------
        prefix : np.ndarray, shape (S,), dtype int64
            Cumulative XOR difference count (prefix[0] should typically be 0
            or the count at the first site).
        positions : np.ndarray, shape (S,), dtype float64
            Physical positions of segregating sites.
        penalty : float or None
            BIC penalty. Default: log(S).

        Returns
        -------
        segments : list of (start_idx, end_idx, tmrca_mle, n_mutations)
        """
        S = len(prefix)
        if S == 0:
            return []

        if penalty is None:
            penalty = np.log(S)

        F = np.full(S, np.inf)
        last_cp = np.zeros(S, dtype=int)
        R = [0]
        F[0] = 0.0
        last_cp[0] = 0

        for s in range(1, S):
            best_cost = np.inf
            best_t = 0

            for t in R:
                count = int(prefix[s] - prefix[t])
                length = positions[s] - positions[t]
                c = self.segment_cost(count, length)
                total = F[t] + c + penalty
                if total < best_cost:
                    best_cost = total
                    best_t = t

            F[s] = best_cost
            last_cp[s] = best_t

            R_new = []
            for t in R:
                count = int(prefix[s] - prefix[t])
                length = positions[s] - positions[t]
                c = self.segment_cost(count, length)
                if F[t] + c <= F[s]:
                    R_new.append(t)
            R_new.append(s)
            R = R_new

        # Traceback
        changepoints = []
        pos = S - 1
        while pos > 0:
            changepoints.append(pos)
            pos = last_cp[pos]
        changepoints.append(0)
        changepoints.reverse()

        segments = []
        for i in range(len(changepoints) - 1):
            seg_start = changepoints[i]
            seg_end = changepoints[i + 1]
            count = int(prefix[seg_end] - prefix[seg_start])
            length = positions[seg_end] - positions[seg_start]
            if count == 0 or length <= 0.0:
                tmrca_mle = 0.0
            else:
                tmrca_mle = count / (2.0 * self.mu * length)
            segments.append((seg_start, seg_end, tmrca_mle, count))

        return segments
