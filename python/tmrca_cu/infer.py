"""Top-level infer() function for tmrca_cu."""

def infer(G_or_ts, positions=None, mu=1.25e-8, rho=1e-8, Ne=10000,
          pairs=None, flow_field_path=None, mean_only=True):
    """Estimate pairwise TMRCA at every segregating site.

    Parameters
    ----------
    G_or_ts : array-like or tskit.TreeSequence
        Either an (n_haplotypes, n_sites) uint8 genotype matrix,
        or a tskit TreeSequence.
    positions : array-like, optional
        Physical positions (bp) of segregating sites. Required if G_or_ts
        is a matrix. Ignored if G_or_ts is a TreeSequence.
    mu : float
        Per-site per-generation mutation rate.
    rho : float
        Per-site per-generation recombination rate.
    Ne : float
        Effective population size.
    pairs : list of (int, int), optional
        Pairs of haplotype indices. Default: all pairs.
    flow_field_path : str, optional
        Path to flow field file. Default: bundled default.
    mean_only : bool
        If True, return only posterior mean. If False, also return
        lower/upper 95% credible intervals.

    Returns
    -------
    dict with keys:
        "mean" : ndarray, shape (n_sites, n_pairs)
        "lower", "upper" : ndarray (only if mean_only=False)
        "pairs" : list of (int, int)
        "positions" : ndarray
    """
    import numpy as np
    from tmrca_cu import _core

    # Handle tree sequence input
    if hasattr(G_or_ts, 'genotype_matrix'):
        ts = G_or_ts
        G = ts.genotype_matrix().T.astype(np.uint8)
        positions = np.array([v.position for v in ts.variants()], dtype=np.float64)
    else:
        G = np.ascontiguousarray(G_or_ts, dtype=np.uint8)
        positions = np.ascontiguousarray(positions, dtype=np.float64)

    n = G.shape[0]

    # Default: all pairs
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i)]

    # Default flow field
    if flow_field_path is None:
        import os
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), 'default_flow_field.txt'),
            '/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt',
        ]
        for c in candidates:
            if os.path.exists(c):
                flow_field_path = c
                break
        if flow_field_path is None:
            raise FileNotFoundError(
                "No flow field file found. Pass flow_field_path explicitly.")

    ctx = _core.FlowContext(G, positions, float(Ne), mu, rho,
                             flow_field_path, 0)
    result = ctx.run_fb(pairs, mean_only=mean_only)
    result["pairs"] = pairs
    result["positions"] = positions
    return result
