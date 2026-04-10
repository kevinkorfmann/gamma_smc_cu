"""Top-level inference helpers for tmrca_cu."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np

# Per-pair, per-padded-site GPU memory consumed by the blockwise FB kernel:
#   - forward buffer (mean + cv) -> 2 floats
#   - output arrays              -> 1 float (mean only) or 3 floats (with CI)
_BYTES_PER_FLOAT32 = 4
_MIN_BLOCK_SITES = 1024
_MAX_BLOCK_SITES = 32768
_DEFAULT_BLOCK_SITES = 8192


def _coerce_inputs(G_or_ts, positions):
    """Normalize tree-sequence and matrix inputs to contiguous arrays."""
    if hasattr(G_or_ts, "genotype_matrix"):
        ts = G_or_ts
        G = ts.genotype_matrix().T.astype(np.uint8, copy=False)
        positions_arr = np.array(
            [variant.position for variant in ts.variants()],
            dtype=np.float64,
        )
    else:
        if positions is None:
            raise ValueError("positions is required when G_or_ts is a genotype matrix.")
        G = np.ascontiguousarray(G_or_ts, dtype=np.uint8)
        if G.ndim != 2:
            raise ValueError("G_or_ts must be a 2D haplotype matrix or a tree sequence.")
        positions_arr = np.ascontiguousarray(positions, dtype=np.float64)
        if positions_arr.ndim != 1:
            raise ValueError("positions must be a 1D array of site coordinates.")
        if G.shape[1] != positions_arr.shape[0]:
            raise ValueError("positions length must match the number of sites in G_or_ts.")

    return G, positions_arr


def _estimate_scaled_params(G, positions, mu, rho, Ne):
    """Estimate ``(effective_mu, effective_rho)`` matching gamma_smc's
    auto-estimation mode (Schweiger and Durbin, 2023).

    The gamma_smc binary, when invoked without ``-m``, sets its scaled
    mutation rate to the observed **per-individual** heterozygosity:
    for each diploid individual it counts sites where the two
    haplotypes differ, divides by the sequence length, and averages
    across individuals (``calculate_heterozygosity()`` in
    ``data_processor.h``). Under the standard coalescent this is an
    unbiased estimator of ``4*Ne*mu``, so it effectively learns the
    *data-implied* effective Ne instead of blindly trusting the user's
    ``Ne`` constant. The scaled recombination rate is then derived
    from the user-supplied ``rho/mu`` ratio.

    We use the per-individual formula (rather than pairwise pi over all
    n*(n-1)/2 pairs) so that ``tmrca_cu.infer()`` and the gamma_smc
    binary see the *same* numeric value for ``scaled_mu`` on every
    input. Empirically the two estimators agree to four decimals on
    HomSap but differ by up to 4% on plant / mosquito models with
    very dense segregating sites — that small difference matters for
    the lowest-rho/mu configs (e.g. ``AraTha African2Epoch_1H18``).

    tmrca.cu's ``FlowContext`` now matches gamma_smc's ``4*Ne``
    convention for both scaled rates, so we return ``effective_mu`` and
    ``effective_rho`` with the inverses baked in:
    ``4*Ne*eff_mu == pi_hat`` and
    ``4*Ne*eff_rho == pi_hat * (rho/mu)``.

    Returns the original ``(mu, rho)`` unchanged if the input has
    fewer than two haplotypes, an odd number of haplotypes (unphased
    diploid layout impossible), zero sites, or non-positive pi_hat.
    """
    n, S = G.shape
    if n < 2 or S == 0 or n % 2 != 0:
        return float(mu), float(rho)
    # gamma_smc's default no-mask path uses a global mask spanning
    # [0, last_pos + 1), so the heterozygosity denominator is the full
    # left-anchored contig span implied by the VCF coordinates rather than
    # the distance between the first and last segregating sites.
    seq_len = float(positions[-1] + 1.0)
    if seq_len <= 0:
        return float(mu), float(rho)
    nd = n // 2
    # XOR within each diploid pair, summed across sites, then mean
    # across individuals (Schweiger & Durbin 2023, data_processor.h:182).
    h0 = G[0::2]                       # (nd, S)  haplotype-0 of each diploid
    h1 = G[1::2]                       # (nd, S)  haplotype-1 of each diploid
    het_counts = (h0 != h1).sum(axis=1, dtype=np.int64)   # (nd,)
    pi_hat = float(het_counts.mean()) / seq_len
    if not (pi_hat > 0 and math.isfinite(pi_hat)):
        return float(mu), float(rho)
    ratio = float(rho) / max(float(mu), 1e-30)
    effective_mu = pi_hat / (4.0 * float(Ne))
    effective_rho = pi_hat * ratio / (4.0 * float(Ne))
    return float(effective_mu), float(effective_rho)


def _filter_segregating(G, positions):
    """Drop sites that are monomorphic within ``G``.

    Gamma-SMC's observation model operates on segregating sites only;
    the homozygous stretches between segregating sites are handled
    analytically by the per-bp mutation emission (``beta += 2*mu*gap``
    in the forward pass). Feeding monomorphic-in-subset sites to the
    kernel would add extra moment-match transition steps outside the
    algorithm's validated envelope — the original gamma_smc reference
    implementation filters these out at VCF parse time
    (``io.h:462``, Schweiger & Durbin, 2023).

    Returns the filtered ``(G, positions)`` plus the boolean keep mask
    so callers can project auxiliary per-site arrays onto the kept
    sites. For already-segregating input (e.g. an msprime tree
    sequence) the filter is a no-op and no copy is made.
    """
    n = G.shape[0]
    ac = G.sum(axis=0, dtype=np.int64)
    keep = (ac > 0) & (ac < n)
    if bool(keep.all()):
        return G, positions, keep
    G_filt = np.ascontiguousarray(G[:, keep])
    positions_filt = np.ascontiguousarray(positions[keep])
    return G_filt, positions_filt, keep


def _resolve_flow_field_path(flow_field_path):
    """Locate the default flow field if the caller did not pass one."""
    if flow_field_path is not None:
        return flow_field_path

    candidates = [
        os.path.join(os.path.dirname(__file__), "default_flow_field.txt"),
        "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No flow field file found. Pass flow_field_path explicitly."
    )


def _normalize_pairs(pairs):
    return [(int(i), int(j)) for i, j in pairs]


def _gpu_bytes_per_pair_site(mean_only):
    """Per-pair, per-padded-site GPU memory in bytes for blockwise FB."""
    n_arrays = 1 if mean_only else 3
    return _BYTES_PER_FLOAT32 * (2 + n_arrays)


def _query_free_gpu_bytes():
    """Return free GPU memory in bytes for the active CUDA device, or None."""
    try:
        from tmrca_cu import _core
        free_bytes, _total = _core.cuda_mem_info()
        return int(free_bytes)
    except Exception:
        return None


def _estimate_block_gpu_bytes(core_block_sites, flank_sites, n_pairs, mean_only,
                              n_sites=None):
    """Estimated GPU bytes for one blockwise FB pass with all pairs in one batch.

    The C++ backend clamps every padded block to ``[0, n_sites]``, so when
    ``n_sites`` is provided we clamp the estimate the same way (otherwise
    tiny inputs would trigger spurious warnings).
    """
    padded = core_block_sites + 2 * flank_sites
    if n_sites is not None:
        padded = min(padded, n_sites)
    return _gpu_bytes_per_pair_site(mean_only) * padded * n_pairs


def _recommend_core_block_size(
    n_sites,
    n_pairs,
    mean_only,
    flank_sites,
    free_bytes,
    headroom_fraction=0.20,
    min_headroom_bytes=512 * 1024 * 1024,
    min_block=_MIN_BLOCK_SITES,
    max_block=_MAX_BLOCK_SITES,
):
    """Pick the largest core_block_sites that fits all pairs in one batch.

    Returns the chosen core_block_sites (clamped to [min_block, n_sites]) and
    its estimated GPU memory footprint in bytes.
    """
    if n_pairs <= 0:
        chosen = min(n_sites, max_block)
        return chosen, 0
    bps = _gpu_bytes_per_pair_site(mean_only)
    headroom = max(min_headroom_bytes, int(free_bytes * headroom_fraction))
    budget = max(0, free_bytes - headroom)
    padded_max = budget // (bps * n_pairs)
    core_max = padded_max - 2 * flank_sites
    if core_max <= 0:
        chosen = min(n_sites, min_block)
    elif core_max >= n_sites:
        chosen = n_sites
    else:
        # Round down to a power of two for cache friendliness, then clamp.
        nice = 1 << int(math.log2(core_max))
        chosen = max(min_block, min(nice, max_block, n_sites))
    estimate = _estimate_block_gpu_bytes(chosen, flank_sites, n_pairs, mean_only)
    return chosen, estimate


def infer(
    G_or_ts,
    positions=None,
    mu=1.25e-8,
    rho=1e-8,
    Ne=10000,
    pairs=None,
    flow_field_path=None,
    mean_only=True,
    return_posterior=False,
    auto_estimate_theta=True,
):
    """Estimate pairwise TMRCA at every segregating site.

    Monomorphic sites (allele count 0 or equal to ``n_haplotypes``) in
    the provided ``G`` are silently dropped before decoding, matching
    the original gamma_smc reference implementation (Schweiger &
    Durbin, 2023), which filters at VCF parse time. The returned
    ``result["positions"]`` therefore reflects the segregating sites
    actually decoded and may be shorter than the input ``positions``.
    For inputs that are already segregating-only (e.g. an msprime tree
    sequence) the filter is a no-op.

    Parameters
    ----------
    mean_only : bool, default True
        If False, also return Wilson-Hilferty 95% CI bounds as ``lower``
        and ``upper`` arrays.
    return_posterior : bool, default False
        If True, also return the per-site combined Gamma posterior
        parameters as ``posterior_alpha`` and ``posterior_beta`` arrays
        in scaled coalescent time (T_scaled = T / (2*Ne)). Mean in
        generations is then ``(alpha / beta) * 2 * Ne``; arbitrary
        quantiles can be computed via ``scipy.stats.gamma(alpha,
        scale=2*Ne/beta).ppf(q)``.
    auto_estimate_theta : bool, default True
        If True (the default), replace the scaled parameters derived
        from ``(Ne, mu, rho)`` with ones learned from the data, matching
        gamma_smc's auto-estimation mode (``calculate_heterozygosity()``
        plus ``-t rho/mu``). The scaled mutation rate is set to the
        observed per-individual heterozygosity and the scaled
        recombination rate to ``pi_hat * (rho/mu)``; ``Ne`` is only used
        to invert tmrca.cu's internal per-bp scaling. This consistently outperforms the
        naive ``Ne=10000`` assumption on non-HomSap species where the
        data-implied effective Ne differs from the user-supplied one.
        Pass ``False`` to force the kernel to use the raw
        ``(Ne, mu, rho)`` values — useful for demographic
        misspecification studies.
    """
    from tmrca_cu import _core

    G, positions = _coerce_inputs(G_or_ts, positions)
    G, positions, _ = _filter_segregating(G, positions)
    n = G.shape[0]

    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i)]
    else:
        pairs = _normalize_pairs(pairs)

    flow_field_path = _resolve_flow_field_path(flow_field_path)

    if auto_estimate_theta:
        kernel_mu, kernel_rho = _estimate_scaled_params(
            G, positions, mu, rho, Ne
        )
    else:
        kernel_mu, kernel_rho = float(mu), float(rho)

    ctx = _core.FlowContext(
        G, positions, float(Ne), kernel_mu, kernel_rho, flow_field_path, 0
    )
    result = ctx.run_fb(
        pairs,
        mean_only=mean_only,
        return_posterior=return_posterior,
    )
    result["pairs"] = pairs
    result["positions"] = positions
    return result


def infer_blockwise(
    G_or_ts,
    positions=None,
    mu=1.25e-8,
    rho=1e-8,
    Ne=10000,
    pairs=None,
    flow_field_path=None,
    mean_only=True,
    core_block_sites="auto",
    flank_sites=2048,
    pair_batch_size=-1,
    max_streams=1,
    verbose=False,
    return_posterior=False,
    auto_estimate_theta=True,
):
    """Blockwise Gamma-SMC forward-backward decoding for explicit pairs.

    Decodes the sequence in chunks of ``core_block_sites`` sites, each padded
    with ``flank_sites`` sites of context on either side, then stitches the
    core regions of every block back into the usual ``(n_sites, n_pairs)``
    site-major output array. The flanks act as a forward/backward burn-in
    so the per-block posterior matches the full-sequence posterior to
    floating-point precision.

    Monomorphic sites in ``G`` are filtered before decoding to match
    :func:`infer` and the original gamma_smc (Schweiger & Durbin,
    2023); see :func:`infer` for details.

    Memory: peak GPU usage is bounded by one padded block,
    ``(core_block_sites + 2 * flank_sites) * pair_chunk * (12 or 20) bytes``,
    instead of the full ``n_sites * n_pairs * ...`` forward buffer that
    :func:`infer` allocates. This is the entire point of the function and
    is what makes cohort-sized inputs feasible.

    The defaults are chosen so that ``infer_blockwise(G, pos, pairs=pairs)``
    just works on any GPU: free VRAM is queried, the largest core block that
    fits all pairs in a single batch is selected, and the C++ backend further
    auto-chunks pairs if even that doesn't fit. On a GPU with plenty of
    headroom this collapses to a single full-sequence block — i.e. it
    behaves exactly like :func:`infer` but without the up-front allocation.

    Parameters
    ----------
    core_block_sites : int or "auto", default "auto"
        Sites kept from each block. ``"auto"`` queries free GPU memory and
        picks the largest power-of-two block that fits all pairs in a single
        batch (clamped to ``[1024, 32768]``, or to ``n_sites`` if everything
        fits in one block). Pass an int to override.
    flank_sites : int, default 2048
        Burn-in sites on either side of the core. Must be > 0 whenever
        the sequence is split into more than one block, otherwise per-block
        forward/backward passes start from the wrong prior and the result
        is garbage. Rejected at the wrapper level.
    pair_batch_size : int, default -1
        Maximum pairs per kernel launch. ``-1`` lets the C++ backend
        auto-chunk pairs to fit available GPU memory. Pass a positive int
        to cap the chunk size manually.
    max_streams : int, default 1
        Number of concurrent CUDA streams. ``2`` typically halves wall time
        for large inputs at the cost of ~2x peak GPU memory.
    verbose : bool, default False
        If True, print the chosen block sizing and memory estimate.
    return_posterior : bool, default False
        If True, also return the per-site combined Gamma posterior
        parameters as ``posterior_alpha`` and ``posterior_beta`` arrays.
        Currently supported only with ``max_streams=1``.
    """
    from tmrca_cu import _core

    # ----- argument validation that doesn't require knowing n_sites -----
    if pairs is None:
        raise ValueError("infer_blockwise requires explicit pairs in v1.")
    if return_posterior and max_streams > 1:
        raise ValueError(
            "infer_blockwise: return_posterior=True is currently only supported "
            "with max_streams=1. Run blockwise posterior decoding in single-stream "
            "mode."
        )
    if flank_sites < 0:
        raise ValueError("flank_sites must be non-negative.")
    if pair_batch_size == 0 or pair_batch_size < -1:
        raise ValueError("pair_batch_size must be positive or -1 for auto.")
    if max_streams <= 0:
        raise ValueError("max_streams must be positive.")
    auto_block = core_block_sites in ("auto", None)
    if not auto_block:
        if not isinstance(core_block_sites, (int, np.integer)):
            raise TypeError(
                f"core_block_sites must be int or 'auto', got {type(core_block_sites).__name__}."
            )
        if core_block_sites <= 0:
            raise ValueError("core_block_sites must be positive (or 'auto').")

    # ----- parse inputs to learn n_sites / n_pairs -----
    G, positions = _coerce_inputs(G_or_ts, positions)
    G, positions, _ = _filter_segregating(G, positions)
    pairs = _normalize_pairs(pairs)
    n_sites = G.shape[1]
    n_pairs = len(pairs)

    # ----- auto-size or sanity-check core_block_sites -----
    free_bytes = _query_free_gpu_bytes()
    if auto_block:
        if free_bytes is None:
            warnings.warn(
                "infer_blockwise(core_block_sites='auto') could not query GPU "
                f"memory; falling back to core_block_sites={_DEFAULT_BLOCK_SITES}.",
                stacklevel=2,
            )
            core_block_sites = _DEFAULT_BLOCK_SITES
        else:
            core_block_sites, est_bytes = _recommend_core_block_size(
                n_sites, n_pairs, mean_only, flank_sites, free_bytes,
            )
            if verbose:
                print(
                    f"[infer_blockwise] auto core_block_sites={core_block_sites} "
                    f"(free_gpu={free_bytes / 1e9:.2f}GB, "
                    f"est_block={est_bytes / 1e9:.2f}GB, "
                    f"n_blocks={max(1, math.ceil(n_sites / core_block_sites))})",
                    flush=True,
                )
    elif free_bytes is not None and n_pairs > 0:
        est_bytes = _estimate_block_gpu_bytes(
            core_block_sites, flank_sites, n_pairs, mean_only, n_sites=n_sites,
        )
        if est_bytes > free_bytes:
            recommended, _ = _recommend_core_block_size(
                n_sites, n_pairs, mean_only, flank_sites, free_bytes,
            )
            warnings.warn(
                f"infer_blockwise: core_block_sites={core_block_sites} with "
                f"flank_sites={flank_sites} for {n_pairs} pairs needs "
                f"~{est_bytes / 1e9:.2f}GB GPU memory but only "
                f"{free_bytes / 1e9:.2f}GB is free. The C++ backend will fall "
                f"back to chunking pairs (slower). Suggested: "
                f"core_block_sites={recommended} (or pass 'auto').",
                stacklevel=2,
            )

    # ----- correctness gate: multi-block decoding requires flank > 0 -----
    if flank_sites == 0 and core_block_sites < n_sites:
        raise ValueError(
            "infer_blockwise: flank_sites=0 with core_block_sites < n_sites "
            f"(core_block_sites={core_block_sites}, n_sites={n_sites}) "
            "produces invalid results because each block restarts the "
            "forward/backward pass from the marginal prior with no warm-up "
            "history. Pass flank_sites>=256 (default 2048) or set "
            "core_block_sites>=n_sites to decode the whole sequence as a "
            "single block."
        )

    # The C++ backend allocates per-block GPU buffers sized by pair_batch_size
    # *before* it learns how many pairs we actually have. With pair_batch_size=-1
    # the auto-cap can be much larger than n_pairs on a roomy GPU, causing huge
    # wasted allocations. Cap by n_pairs here so the buffer matches the work.
    effective_pair_batch_size = pair_batch_size
    if effective_pair_batch_size == -1:
        effective_pair_batch_size = max(1, n_pairs)

    flow_field_path = _resolve_flow_field_path(flow_field_path)

    if auto_estimate_theta:
        kernel_mu, kernel_rho = _estimate_scaled_params(
            G, positions, mu, rho, Ne
        )
    else:
        kernel_mu, kernel_rho = float(mu), float(rho)

    ctx = _core.FlowContext(
        G, positions, float(Ne), kernel_mu, kernel_rho, flow_field_path, 0
    )
    result = ctx.run_fb_blockwise(
        pairs,
        core_block_sites=int(core_block_sites),
        flank_sites=flank_sites,
        pair_batch_size=effective_pair_batch_size,
        max_streams=max_streams,
        mean_only=mean_only,
        return_posterior=return_posterior,
    )
    result["pairs"] = pairs
    result["positions"] = positions
    return result
