import pytest
import numpy as np
import msprime


@pytest.fixture(scope="session")
def small_simulation():
    """
    Small tree sequence for fast unit tests.
    n=20 haplotypes, 100 kb, constant Ne=10000.
    """
    ts = msprime.sim_ancestry(
        samples=10,  # 10 diploid = 20 haploid
        sequence_length=100_000,
        recombination_rate=1e-8,
        population_size=10_000,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
    G = ts.genotype_matrix().T.astype(np.uint8)  # (n_haplotypes, n_sites)
    positions = np.array([v.position for v in ts.variants()])
    return ts, G, positions


@pytest.fixture(scope="session")
def medium_simulation():
    """
    Medium tree sequence for integration tests.
    n=200 haplotypes, 1 Mb, bottleneck demography.
    """
    demography = msprime.Demography()
    demography.add_population(initial_size=10_000)
    demography.add_population_parameters_change(time=5000, initial_size=1000)
    demography.add_population_parameters_change(time=6000, initial_size=50_000)

    ts = msprime.sim_ancestry(
        samples=100,
        sequence_length=1_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=44,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=45)
    G = ts.genotype_matrix().T
    positions = np.array([v.position for v in ts.variants()])
    return ts, G, positions


@pytest.fixture
def true_pairwise_tmrca(small_simulation):
    """
    Extract true pairwise TMRCA from tree sequence for a set of test pairs.
    """
    ts = small_simulation[0]
    pairs = [(0, 1), (0, 5), (1, 2), (0, 19)]
    result = {}
    for i, j in pairs:
        tmrca = np.zeros(int(ts.sequence_length))
        for tree in ts.trees():
            left, right = int(tree.interval.left), int(tree.interval.right)
            tmrca[left:right] = tree.tmrca(i, j)
        result[(i, j)] = tmrca
    return result


@pytest.fixture
def uniform_mu():
    return 1.25e-8


@pytest.fixture
def uniform_rho():
    return 1e-8
