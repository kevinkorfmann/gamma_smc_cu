#pragma once

#include <cstdint>

// Maximum segments per pair (compile-time upper bound)
#define PELT_MAX_SEGMENTS 1024

/**
 * PELT (Pruned Exact Linear Time) changepoint detection on GPU.
 *
 * Each pair runs an independent PELT instance.  One warp (32 threads)
 * is assigned per pair; threads collaborate on evaluating candidate
 * changepoints via warp-level reductions.
 *
 * The Poisson segment cost for [a, b) is:
 *   cost(a, b) = 2 * mu * L           if count == 0
 *              = -C * log(C / L) + C   otherwise
 * where C = prefix[b] - prefix[a] and L = positions[b] - positions[a].
 *
 * Penalty defaults to BIC: beta = log(S).
 *
 * Parameters
 * ----------
 * prefix       : [n_pairs * S]  cumulative XOR difference count per pair
 * positions    : [S]            physical positions of segregating sites
 * S            : number of segregating sites
 * n_pairs      : number of pairs
 * mu           : per-bp per-generation mutation rate
 * penalty      : BIC penalty per changepoint (typically log(S))
 * n_segments_out  : [n_pairs]                 number of segments found
 * seg_starts_out  : [n_pairs * max_segments]  segment start site indices
 * seg_ends_out    : [n_pairs * max_segments]  segment end site indices
 * seg_tmrca_out   : [n_pairs * max_segments]  MLE TMRCA per segment
 * seg_counts_out  : [n_pairs * max_segments]  mutation count per segment
 * max_segments    : maximum segments per pair (controls output buffer size)
 */
void pelt_changepoint_gpu(
    const int64_t* prefix,
    const double* positions,
    int S, int n_pairs,
    double mu, double penalty,
    int* n_segments_out,
    int* seg_starts_out,
    int* seg_ends_out,
    float* seg_tmrca_out,
    int* seg_counts_out,
    int max_segments
);
