"""
Pure-numpy reference implementation of ultrametric projection via agglomerative clustering.
Used for validating the CUDA ultrametric kernel.
"""

import numpy as np
from itertools import combinations


def _pair_index(i, j):
    """Map ordered pair (i, j) with i > j to linear index."""
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j


class NumpyUltrametric:
    """Reference ultrametric projection for a subsample of haplotypes at one site."""

    def __init__(self, K=32, Ne=10_000, t_max=None):
        """
        Parameters
        ----------
        K : int
            Number of discrete time bins.
        Ne : float
            Effective population size (constant).
        t_max : float or None
            Maximum coalescence time. Defaults to 10 * Ne.
        """
        self.K = K
        self.Ne = Ne
        if t_max is None:
            t_max = 10.0 * Ne
        self.t_max = t_max
        self.time_boundaries = np.array(
            [t_max * (k / K) ** 2 for k in range(K + 1)]
        )
        self.time_midpoints = (self.time_boundaries[:-1] + self.time_boundaries[1:]) / 2.0

        # Coalescent prior
        self.coal_prior = self._compute_prior()

    def _compute_prior(self):
        """Coalescent prior q[k] under constant Ne."""
        q = np.zeros(self.K)
        for k in range(self.K):
            t_lo = self.time_boundaries[k]
            t_hi = self.time_boundaries[k + 1]
            q[k] = np.exp(-t_lo / (2.0 * self.Ne)) - np.exp(-t_hi / (2.0 * self.Ne))
        q_sum = q.sum()
        if q_sum > 0:
            q /= q_sum
        return q

    def project(self, posteriors, m):
        """
        Given pairwise posteriors, find the ultrametric-consistent MAP assignment
        via agglomerative clustering.

        Parameters
        ----------
        posteriors : dict of (i, j) -> np.ndarray, shape (K,)
            Pairwise posterior marginals for all pairs in the subsample.
            Keys are (i, j) with i < j, values are posterior vectors.
        m : int
            Number of haplotypes in the subsample.

        Returns
        -------
        assignment : dict of (i, j) -> int
            Assigned time bin index for each pair.
        """
        # Initialize: each haplotype is its own cluster
        # cluster_members[c] = set of haplotype indices in cluster c
        cluster_members = {i: {i} for i in range(m)}
        active = set(range(m))

        # Track the maximum internal merge time for each cluster
        # (singletons have no internal merges, so start at -1)
        cluster_max_time = {i: -1 for i in range(m)}

        # Store assigned merge times: when clusters containing i and j merge
        # We'll record assignments as we go
        pair_assignment = {}

        for merge_step in range(m - 1):
            best_score = -np.inf
            best_A = -1
            best_B = -1
            best_k = -1

            active_list = sorted(active)

            for idx_a in range(len(active_list)):
                for idx_b in range(idx_a + 1, len(active_list)):
                    A = active_list[idx_a]
                    B = active_list[idx_b]

                    # Ultrametric constraint: merge time must be >= max internal
                    # merge time of both clusters
                    k_min = max(cluster_max_time[A], cluster_max_time[B])
                    if k_min < 0:
                        k_min = 0

                    # For each candidate time bin k, compute score
                    for k in range(k_min, self.K):
                        score = 0.0
                        for i in cluster_members[A]:
                            for j in cluster_members[B]:
                                key = (min(i, j), max(i, j))
                                post = posteriors[key]
                                val = post[k]
                                if val > 0:
                                    score += np.log(val)
                                else:
                                    score += -1e30
                        # Prefer higher score; break ties by lower time bin
                        # (merge earlier when possible)
                        if (score > best_score or
                            (score == best_score and k < best_k)):
                            best_score = score
                            best_A = A
                            best_B = B
                            best_k = k

            # Assign time bin to all cross-cluster pairs
            for i in cluster_members[best_A]:
                for j in cluster_members[best_B]:
                    key = (min(i, j), max(i, j))
                    pair_assignment[key] = best_k

            # Merge: absorb B into A
            cluster_members[best_A] = cluster_members[best_A] | cluster_members[best_B]
            del cluster_members[best_B]
            active.remove(best_B)

            # Update max internal merge time for the merged cluster
            cluster_max_time[best_A] = best_k
            del cluster_max_time[best_B]

        return pair_assignment

    def compute_messages(self, posteriors, m, damping=0.5):
        """
        Run ultrametric projection and compute updated EP messages.

        Parameters
        ----------
        posteriors : dict of (i, j) -> np.ndarray, shape (K,)
        m : int
        damping : float
            Blending factor alpha: msg = alpha * delta(assigned) + (1-alpha) * prior.

        Returns
        -------
        messages : dict of (i, j) -> np.ndarray, shape (K,)
            Updated messages for each pair.
        assignment : dict of (i, j) -> int
            Assigned time bin for each pair.
        """
        assignment = self.project(posteriors, m)
        messages = {}

        for key, assigned_k in assignment.items():
            msg = np.zeros(self.K)
            for k in range(self.K):
                delta = 1.0 if k == assigned_k else 0.0
                msg[k] = damping * delta + (1.0 - damping) * self.coal_prior[k]
            # Normalize
            msg_sum = msg.sum()
            if msg_sum > 0:
                msg /= msg_sum
            messages[key] = msg

        return messages, assignment

    def generate_noisy_tree(self, m, noise_level=0.3, seed=None):
        """
        Generate a random ultrametric tree and noisy posteriors for testing.

        Parameters
        ----------
        m : int
            Number of haplotypes (leaves).
        noise_level : float
            Controls how much noise is added to the true posteriors.
            0 = perfect delta functions, 1 = nearly uniform.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        posteriors : dict of (i, j) -> np.ndarray, shape (K,)
            Noisy posterior marginals.
        true_assignment : dict of (i, j) -> int
            True time bin assignment from the generated tree.
        """
        rng = np.random.RandomState(seed)

        # Generate a random ultrametric tree by agglomerative process
        # Start with m singleton clusters, merge two random clusters at
        # increasing time bins
        cluster_members = {i: {i} for i in range(m)}
        active = list(range(m))
        true_assignment = {}

        # Choose m-1 DISTINCT merge times in strictly increasing order
        # This ensures a fully resolved binary tree (no polytomies),
        # which the greedy agglomerative projection can always recover.
        available_bins = list(range(self.K))
        if m - 1 > len(available_bins):
            # More merges than bins: allow repeats but space them
            merge_bins = sorted(rng.choice(self.K, size=m - 1, replace=True))
        else:
            chosen = sorted(rng.choice(available_bins, size=m - 1, replace=False))
            merge_bins = chosen

        for step in range(m - 1):
            # Pick two random active clusters
            idx_pair = rng.choice(len(active), size=2, replace=False)
            A = active[idx_pair[0]]
            B = active[idx_pair[1]]
            k = merge_bins[step]

            # Assign all cross-cluster pairs
            for i in cluster_members[A]:
                for j in cluster_members[B]:
                    key = (min(i, j), max(i, j))
                    true_assignment[key] = k

            # Merge
            cluster_members[A] = cluster_members[A] | cluster_members[B]
            del cluster_members[B]
            active.remove(B)

        # Generate noisy posteriors
        posteriors = {}
        for i in range(m):
            for j in range(i + 1, m):
                key = (i, j)
                k_true = true_assignment[key]
                # Create a peaked distribution around the true bin
                post = np.ones(self.K) * noise_level / self.K
                post[k_true] += (1.0 - noise_level)
                # Add some random noise
                noise = rng.dirichlet(np.ones(self.K) * 0.1) * noise_level * 0.5
                post += noise
                post /= post.sum()
                posteriors[key] = post

        return posteriors, true_assignment

    def generate_delta_tree(self, m, seed=None):
        """
        Generate a random tree with perfect delta-function posteriors.

        Returns
        -------
        posteriors : dict of (i, j) -> np.ndarray, shape (K,)
        true_assignment : dict of (i, j) -> int
        """
        posteriors, true_assignment = self.generate_noisy_tree(
            m, noise_level=0.0, seed=seed
        )
        # Make truly delta
        for key, k_true in true_assignment.items():
            post = np.zeros(self.K)
            post[k_true] = 1.0
            posteriors[key] = post
        return posteriors, true_assignment

    def ultrametric_violation(self, assignment, m):
        """
        Measure how much a pairwise assignment violates the ultrametric constraint.

        For every triple (i, j, k), the two largest TMRCA values must be equal.

        Parameters
        ----------
        assignment : dict of (i, j) -> int
            Time bin assignment for each pair.
        m : int
            Number of haplotypes.

        Returns
        -------
        violation_count : int
            Number of triples that violate the ultrametric constraint.
        total_triples : int
            Total number of triples checked.
        """
        violation_count = 0
        total_triples = 0

        for a in range(m):
            for b in range(a + 1, m):
                for c in range(b + 1, m):
                    t_ab = assignment[(a, b)]
                    t_ac = assignment[(a, c)]
                    t_bc = assignment[(b, c)]

                    vals = sorted([t_ab, t_ac, t_bc])
                    # Ultrametric: two largest must be equal
                    if vals[1] != vals[2]:
                        violation_count += 1
                    total_triples += 1

        return violation_count, total_triples

    def map_assignment(self, posteriors, m):
        """
        Compute per-pair MAP assignment (independent, not ultrametric-consistent).

        Parameters
        ----------
        posteriors : dict of (i, j) -> np.ndarray, shape (K,)
        m : int

        Returns
        -------
        assignment : dict of (i, j) -> int
        """
        assignment = {}
        for i in range(m):
            for j in range(i + 1, m):
                key = (i, j)
                assignment[key] = int(np.argmax(posteriors[key]))
        return assignment
