"""
Unit tests for ultrametric projection.
Tests the pure-numpy reference implementation directly.
GPU kernel tests are conditional on CUDA availability.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.reference.ultrametric_numpy import NumpyUltrametric


class TestUltrametricProjection:
    """Test the ultrametric projection algorithm."""

    @pytest.fixture
    def ultra(self):
        return NumpyUltrametric(K=32, Ne=10_000)

    @pytest.fixture
    def ultra_small(self):
        """Smaller K for faster combinatorial tests."""
        return NumpyUltrametric(K=8, Ne=10_000)

    def test_already_ultrametric(self, ultra):
        """
        Delta-function posteriors on a valid ultrametric tree should produce
        an assignment identical to the input MAP.
        """
        m = 6
        posteriors, true_assignment = ultra.generate_delta_tree(m, seed=42)

        result = ultra.project(posteriors, m)

        for key in true_assignment:
            assert result[key] == true_assignment[key], (
                f"Pair {key}: expected bin {true_assignment[key]}, got {result[key]}"
            )

    def test_output_is_valid_ultrametric(self, ultra_small):
        """
        The output of ultrametric projection must satisfy the ultrametric
        constraint: for any triple (i, j, k), the two largest TMRCA values
        must be equal.
        """
        m = 8
        # Use noisy posteriors that are NOT ultrametric
        posteriors, _ = ultra_small.generate_noisy_tree(m, noise_level=0.5, seed=123)

        assignment = ultra_small.project(posteriors, m)

        violations, total = ultra_small.ultrametric_violation(assignment, m)
        assert violations == 0, (
            f"Ultrametric violated in {violations}/{total} triples"
        )

    def test_output_is_valid_ultrametric_large(self, ultra):
        """Same ultrametric validity test with m=20, K=32."""
        m = 20
        posteriors, _ = ultra.generate_noisy_tree(m, noise_level=0.4, seed=99)

        assignment = ultra.project(posteriors, m)

        violations, total = ultra.ultrametric_violation(assignment, m)
        assert violations == 0, (
            f"Ultrametric violated in {violations}/{total} triples"
        )

    def test_denoising_effect(self, ultra_small):
        """
        Given noisy posteriors from a known tree, the ultrametric projection
        should produce a valid ultrametric result. We also check that the
        average time-bin error is reasonable (not wildly off from truth).
        """
        m = 6
        posteriors, true_assignment = ultra_small.generate_noisy_tree(
            m, noise_level=0.3, seed=77
        )

        # Independent MAP (no ultrametric constraint)
        map_assignment = ultra_small.map_assignment(posteriors, m)

        # Ultrametric projection
        ultra_assignment = ultra_small.project(posteriors, m)

        # The ultrametric result must be valid (MAP typically is not)
        ultra_viol, _ = ultra_small.ultrametric_violation(ultra_assignment, m)
        assert ultra_viol == 0, "Ultrametric projection must produce valid ultrametric"

        map_viol, _ = ultra_small.ultrametric_violation(map_assignment, m)

        # The MAP is often not ultrametric, while our projection always is
        # This is the key advantage of the projection
        if map_viol > 0:
            assert ultra_viol < map_viol, (
                "Projection should have fewer violations than unconstrained MAP"
            )

        # Check that the ultrametric assignment is reasonably close to truth
        # (mean absolute bin error should be bounded)
        n_pairs = m * (m - 1) // 2
        ultra_errors = [
            abs(ultra_assignment[key] - true_assignment[key])
            for key in true_assignment
        ]
        mean_error = np.mean(ultra_errors)
        assert mean_error < ultra_small.K / 2, (
            f"Mean bin error {mean_error:.1f} too large (K={ultra_small.K})"
        )

    def test_single_pair(self, ultra):
        """
        m=2: trivial case with only one pair. The projection should return
        the MAP bin (argmax of the posterior).
        """
        m = 2
        K = ultra.K

        # Create a simple posterior peaked at bin 10
        post = np.zeros(K)
        post[10] = 0.8
        post[9] = 0.1
        post[11] = 0.1
        posteriors = {(0, 1): post}

        assignment = ultra.project(posteriors, m)

        assert assignment[(0, 1)] == 10, (
            f"Single pair should get MAP bin 10, got {assignment[(0, 1)]}"
        )

    def test_single_pair_uniform(self, ultra):
        """m=2 with uniform posterior: should pick some bin (no crash)."""
        m = 2
        K = ultra.K
        post = np.ones(K) / K
        posteriors = {(0, 1): post}

        assignment = ultra.project(posteriors, m)
        assert 0 <= assignment[(0, 1)] < K

    def test_message_damping(self, ultra):
        """
        Verify that message computation correctly blends between the tree
        assignment (delta function) and the coalescent prior.
        """
        m = 4
        K = ultra.K

        posteriors, true_assignment = ultra.generate_delta_tree(m, seed=55)

        for damping in [0.0, 0.3, 0.5, 0.7, 1.0]:
            messages, assignment = ultra.compute_messages(posteriors, m, damping=damping)

            for key, ak in assignment.items():
                msg = messages[key]

                # Check message is a valid distribution
                assert msg.sum() > 0, "Message sums to zero"
                np.testing.assert_allclose(msg.sum(), 1.0, rtol=1e-6,
                                           err_msg=f"Message not normalized for damping={damping}")

                # Check all values non-negative
                assert np.all(msg >= 0), f"Negative message values for damping={damping}"

                # The assigned bin should have the highest message weight
                # (or tied) for any damping > 0
                if damping > 0:
                    assert msg[ak] >= msg.max() - 1e-7, (
                        f"Assigned bin {ak} not maximal in message for damping={damping}"
                    )

    def test_message_damping_zero(self, ultra):
        """With damping=0, messages should equal the coalescent prior."""
        m = 4
        posteriors, _ = ultra.generate_delta_tree(m, seed=60)
        messages, _ = ultra.compute_messages(posteriors, m, damping=0.0)

        for key, msg in messages.items():
            np.testing.assert_allclose(msg, ultra.coal_prior, rtol=1e-6,
                                       err_msg="damping=0 should give prior")

    def test_message_damping_one(self, ultra):
        """With damping=1, messages should be pure delta at the assigned bin."""
        m = 4
        posteriors, _ = ultra.generate_delta_tree(m, seed=61)
        messages, assignment = ultra.compute_messages(posteriors, m, damping=1.0)

        for key, msg in messages.items():
            expected = np.zeros(ultra.K)
            expected[assignment[key]] = 1.0
            np.testing.assert_allclose(msg, expected, atol=1e-7,
                                       err_msg="damping=1 should give delta")

    def test_three_haplotype_cherry(self, ultra):
        """
        Three haplotypes forming a cherry: (0,1) coalesce first, then
        the ancestor merges with 2. Check that the projection recovers this.
        """
        m = 3
        K = ultra.K
        k_close = 5   # close pair
        k_far = 15    # distant pair

        # (0,1) are close, (0,2) and (1,2) are far
        posteriors = {}
        # Strong signal at k_close for pair (0,1)
        p01 = np.zeros(K)
        p01[k_close] = 0.9
        p01[k_close + 1] = 0.1
        posteriors[(0, 1)] = p01

        # Strong signal at k_far for pairs involving 2
        for pair in [(0, 2), (1, 2)]:
            p = np.zeros(K)
            p[k_far] = 0.9
            p[k_far + 1] = 0.1
            posteriors[pair] = p

        assignment = ultra.project(posteriors, m)

        assert assignment[(0, 1)] == k_close
        assert assignment[(0, 2)] == k_far
        assert assignment[(1, 2)] == k_far

        # Verify ultrametric property
        violations, _ = ultra.ultrametric_violation(assignment, m)
        assert violations == 0

    def test_violation_count_on_non_ultrametric(self, ultra):
        """Verify the violation counter correctly identifies non-ultrametric assignments."""
        m = 3
        # Non-ultrametric: all three pairs have different values
        bad_assignment = {(0, 1): 5, (0, 2): 10, (1, 2): 15}
        violations, total = ultra.ultrametric_violation(bad_assignment, m)
        assert violations == 1
        assert total == 1

        # Ultrametric: two largest are equal
        good_assignment = {(0, 1): 5, (0, 2): 10, (1, 2): 10}
        violations, total = ultra.ultrametric_violation(good_assignment, m)
        assert violations == 0

    def test_reproducibility(self, ultra):
        """Same input should give same output (deterministic algorithm)."""
        m = 8
        posteriors, _ = ultra.generate_noisy_tree(m, noise_level=0.4, seed=200)

        a1 = ultra.project(posteriors, m)
        a2 = ultra.project(posteriors, m)

        for key in a1:
            assert a1[key] == a2[key], f"Non-deterministic at pair {key}"

    def test_all_pairs_assigned(self, ultra):
        """Every pair should receive a valid assignment."""
        m = 10
        posteriors, _ = ultra.generate_noisy_tree(m, noise_level=0.3, seed=300)
        assignment = ultra.project(posteriors, m)

        for i in range(m):
            for j in range(i + 1, m):
                key = (i, j)
                assert key in assignment, f"Pair {key} not assigned"
                assert 0 <= assignment[key] < ultra.K, (
                    f"Pair {key} assigned invalid bin {assignment[key]}"
                )
