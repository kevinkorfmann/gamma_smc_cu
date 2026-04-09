"""Top-level inference helpers for tmrca_cu."""

from __future__ import annotations

import os

import numpy as np


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


def infer(
    G_or_ts,
    positions=None,
    mu=1.25e-8,
    rho=1e-8,
    Ne=10000,
    pairs=None,
    flow_field_path=None,
    mean_only=True,
):
    """Estimate pairwise TMRCA at every segregating site."""
    from tmrca_cu import _core

    G, positions = _coerce_inputs(G_or_ts, positions)
    n = G.shape[0]

    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i)]
    else:
        pairs = _normalize_pairs(pairs)

    flow_field_path = _resolve_flow_field_path(flow_field_path)

    ctx = _core.FlowContext(
        G, positions, float(Ne), mu, rho, flow_field_path, 0
    )
    result = ctx.run_fb(pairs, mean_only=mean_only)
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
    core_block_sites=8192,
    flank_sites=2048,
    pair_batch_size=256,
    max_streams=1,
):
    """Experimental blockwise Gamma-SMC decoding for explicit pairs.

    This v1 path keeps the existing pair/site mean TMRCA output but runs
    forward-backward on padded site blocks and stitches the core block results
    back into the usual site-major output shape.
    """
    from tmrca_cu import _core

    if pairs is None:
        raise ValueError("infer_blockwise requires explicit pairs in v1.")
    if core_block_sites <= 0:
        raise ValueError("core_block_sites must be positive.")
    if flank_sites < 0:
        raise ValueError("flank_sites must be non-negative.")
    if pair_batch_size == 0 or pair_batch_size < -1:
        raise ValueError("pair_batch_size must be positive or -1 for auto.")
    if max_streams <= 0:
        raise ValueError("max_streams must be positive.")

    G, positions = _coerce_inputs(G_or_ts, positions)
    pairs = _normalize_pairs(pairs)
    flow_field_path = _resolve_flow_field_path(flow_field_path)

    ctx = _core.FlowContext(
        G, positions, float(Ne), mu, rho, flow_field_path, 0
    )
    result = ctx.run_fb_blockwise(
        pairs,
        core_block_sites=core_block_sites,
        flank_sites=flank_sites,
        pair_batch_size=pair_batch_size,
        max_streams=max_streams,
        mean_only=mean_only,
    )
    result["pairs"] = pairs
    result["positions"] = positions
    return result
