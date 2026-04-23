import numpy as np
import pytest
import sys
import os

# Add project root to path so we can import gamma_smc_cu and test references
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gamma_smc_cu
from tests.reference.hmm_numpy import NumpyHMM


class TestHMMForwardBackward:
    """
    Verify CUDA HMM against pure-numpy reference implementation.
    """

    @pytest.fixture
    def numpy_hmm(self, small_simulation, uniform_mu, uniform_rho):
        _, G, positions = small_simulation
        return NumpyHMM(G, positions, mu=uniform_mu, rho=uniform_rho, K=32, Ne=10_000)

    def test_posterior_marginals(self, small_simulation, numpy_hmm,
                                 uniform_mu, uniform_rho):
        """Posterior gamma matches numpy reference."""
        _, G, positions = small_simulation
        pair = (0, 1)

        np_gamma = numpy_hmm.posterior(pair)
        cuda_gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, pair, K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))

        np.testing.assert_allclose(cuda_gamma, np_gamma, rtol=1e-2, atol=1e-4)

    def test_posterior_sums_to_one(self, small_simulation,
                                    uniform_mu, uniform_rho):
        """Posterior marginals must sum to 1 at every site."""
        _, G, positions = small_simulation
        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))
        sums = gamma.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-3)

    def test_log_likelihood(self, small_simulation, numpy_hmm,
                             uniform_mu, uniform_rho):
        """Total log-likelihood matches reference."""
        _, G, positions = small_simulation
        pair = (0, 1)

        np_ll = numpy_hmm.log_likelihood(pair)
        cuda_ll = gamma_smc_cu.hmm_log_likelihood(
            G, positions, pair, K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        )

        np.testing.assert_allclose(cuda_ll, np_ll, rtol=1e-2)

    def test_different_pairs_give_different_posteriors(self, small_simulation,
                                                        uniform_mu, uniform_rho):
        """Different pairs should generally produce different posteriors."""
        _, G, positions = small_simulation
        gamma1 = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))
        gamma2 = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 5), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))
        # Should not be identical
        assert not np.allclose(gamma1, gamma2, atol=1e-3)

    def test_posterior_values_are_nonneg(self, small_simulation,
                                          uniform_mu, uniform_rho):
        """All posterior values should be non-negative."""
        _, G, positions = small_simulation
        gamma = np.array(gamma_smc_cu.hmm_posterior(
            G, positions, (0, 1), K=32, Ne=10000.0,
            mu=uniform_mu, rho=uniform_rho
        ))
        assert np.all(gamma >= 0)
