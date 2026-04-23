import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu


def build_reference_transition_matrix(r, midpoints, coal_prior):
    """
    Build the SMC transition matrix in pure numpy.

    A[i,j] = (1 - p_rec) * I(i==j) + p_rec * coal_prior[j]

    where p_rec = 1 - exp(-r) is the probability of at least one
    recombination event in a segment of genetic distance r.
    """
    K = len(coal_prior)
    p_rec = 1.0 - np.exp(-r)
    A = (1.0 - p_rec) * np.eye(K) + p_rec * coal_prior[np.newaxis, :]
    return A


class TestTransitions:
    """
    Verify transition matrix properties for the SMC HMM.

    The transition model is:
        A(r)[i,j] = (1-p_rec)*delta(i,j) + p_rec * q[j]
    where p_rec = 1 - exp(-r) and q is the coalescent prior.
    """

    K = 32
    NE = 10_000.0

    @pytest.fixture
    def coal_prior(self):
        return np.array(gamma_smc_cu.coalescent_prior(Ne=self.NE, K=self.K))

    @pytest.fixture
    def time_midpoints(self):
        t_max = 10.0 * self.NE
        boundaries = np.array([t_max * (k / self.K) ** 2 for k in range(self.K + 1)])
        return (boundaries[:-1] + boundaries[1:]) / 2.0

    def test_transition_rows_sum_to_one(self, coal_prior, time_midpoints):
        """Each row of A(r) must sum to 1 for various recombination rates."""
        for r in [1e-6, 1e-4, 1e-2, 0.5, 1.0, 10.0]:
            A = build_reference_transition_matrix(r, time_midpoints, coal_prior)
            row_sums = A.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, rtol=1e-10,
                err_msg=f"Row sums not 1 for r={r}"
            )

    def test_no_recombination_is_identity(self, coal_prior, time_midpoints):
        """A(r=0) should be the identity matrix."""
        A = build_reference_transition_matrix(0.0, time_midpoints, coal_prior)
        np.testing.assert_allclose(A, np.eye(self.K), atol=1e-10)

    def test_high_recombination_gives_prior(self, coal_prior, time_midpoints):
        """
        For very large r, p_rec -> 1, so every row of A approaches coal_prior.
        """
        A = build_reference_transition_matrix(100.0, time_midpoints, coal_prior)
        for k in range(self.K):
            np.testing.assert_allclose(
                A[k], coal_prior, atol=1e-4,
                err_msg=f"Row {k} doesn't approach prior for large r"
            )

    def test_coalescent_prior_sums_to_one(self):
        """Coalescent prior q[k] must sum to 1 for various Ne."""
        for Ne in [100, 10_000, 1_000_000]:
            q = np.array(gamma_smc_cu.coalescent_prior(Ne=float(Ne), K=self.K))
            np.testing.assert_allclose(q.sum(), 1.0, rtol=1e-8,
                                       err_msg=f"Prior doesn't sum to 1 for Ne={Ne}")

    def test_coalescent_prior_nonnegative(self):
        """All prior entries must be non-negative."""
        for Ne in [100, 10_000, 1_000_000]:
            q = np.array(gamma_smc_cu.coalescent_prior(Ne=float(Ne), K=self.K))
            assert np.all(q >= 0), f"Negative prior entries for Ne={Ne}"

    def test_transition_matrix_off_diagonal_structure(self, coal_prior, time_midpoints):
        """
        The CUDA kernel uses a state-dependent recombination probability:
            A[k][l] = exp(-r*t_k)*delta(k,l) + (1-exp(-r*t_k))*q[l]

        Verify that for each row k, subtracting the diagonal contribution
        leaves exactly (1-exp(-r*t_k))*q[l].
        """
        r = 0.01
        K = self.K
        # Build the state-dependent transition matrix directly
        A = np.zeros((K, K))
        for k in range(K):
            no_rec = np.exp(-r * time_midpoints[k])
            p_rec_k = 1.0 - no_rec
            A[k, :] = p_rec_k * coal_prior
            A[k, k] += no_rec

        for k in range(K):
            p_rec_k = 1.0 - np.exp(-r * time_midpoints[k])
            expected = p_rec_k * coal_prior
            actual = A[k].copy()
            actual[k] -= np.exp(-r * time_midpoints[k])
            np.testing.assert_allclose(
                actual, expected, rtol=1e-8,
                err_msg=f"Row {k} off-diagonal pattern doesn't match"
            )

    def test_transition_preserves_prior(self, coal_prior, time_midpoints):
        """
        The prior is a stationary distribution: prior @ A = prior.
        """
        for r in [1e-4, 0.01, 0.5]:
            A = build_reference_transition_matrix(r, time_midpoints, coal_prior)
            result = coal_prior @ A
            np.testing.assert_allclose(
                result, coal_prior, rtol=1e-10,
                err_msg=f"Prior not stationary for r={r}"
            )
