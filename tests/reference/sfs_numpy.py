"""Pure-numpy SFS computation for reference."""

import numpy as np


def compute_sfs_numpy(G):
    """
    Compute the site frequency spectrum from a genotype matrix.

    Parameters
    ----------
    G : np.ndarray, shape (n, S), dtype uint8
        Haploid genotype matrix.

    Returns
    -------
    sfs : np.ndarray, shape (n+1,)
        SFS where sfs[k] = number of sites with derived allele count k.
    """
    n = G.shape[0]
    allele_counts = G.sum(axis=0)  # sum across haplotypes for each site
    sfs = np.zeros(n + 1, dtype=int)
    for count in allele_counts:
        sfs[int(count)] += 1
    return sfs
