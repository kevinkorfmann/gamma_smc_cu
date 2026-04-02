"""
tmrca_cu: GPU-accelerated pairwise coalescence time estimation.
"""

import os as _os
import sys as _sys

# Add the directory containing _core.so to the path
_module_dir = _os.path.dirname(_os.path.abspath(__file__))
if _module_dir not in _sys.path:
    _sys.path.insert(0, _module_dir)

from tmrca_cu._core import (
    bitpack,
    unpack,
    pairwise_prefix_scan,
    windowed_divergence,
    compute_sfs,
    coalescent_prior,
    hmm_posterior,
    hmm_log_likelihood,
    hmm_posterior_batched,
    time_midpoints,
    time_boundaries,
    site_pi,
    pelt_changepoint,
    adaptive_prior_infer,
)

from tmrca_cu.estimator import (
    CoalescenceEstimator,
    TMRCAResult,
    Segment,
)

__all__ = [
    "bitpack",
    "unpack",
    "pairwise_prefix_scan",
    "windowed_divergence",
    "compute_sfs",
    "coalescent_prior",
    "hmm_posterior",
    "hmm_log_likelihood",
    "hmm_posterior_batched",
    "time_midpoints",
    "time_boundaries",
    "site_pi",
    "pelt_changepoint",
    "adaptive_prior_infer",
    "CoalescenceEstimator",
    "TMRCAResult",
    "Segment",
]
