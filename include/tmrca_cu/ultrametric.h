#pragma once

#include "types.h"

// ============================================================
// Ultrametric Projection
// ============================================================
// For each site, find the tree-consistent TMRCA assignment that
// maximizes the joint posterior across a subsample of haplotypes.
// Uses agglomerative clustering: one block per site.

/**
 * @brief GPU kernel launcher for ultrametric projection.
 *
 * @param gamma          [n_sites × n_subsample_pairs × K] posterior marginals (row-major)
 * @param n_haplotypes   m: number of haplotypes in subsample
 * @param K              number of time bins
 * @param coal_prior     [K] coalescent prior (device pointer)
 * @param damping        alpha: damping factor for message update
 * @param messages_out   [n_sites × n_subsample_pairs × K] updated messages (output)
 * @param assigned_bins_out [n_sites × n_subsample_pairs] assigned time bins (output)
 * @param n_sites        number of sites to process
 * @param n_subsample_pairs  m*(m-1)/2
 */
void ultrametric_project_gpu(
    const float* gamma,
    int n_haplotypes,
    int K,
    const double* coal_prior,
    double damping,
    float* messages_out,
    int* assigned_bins_out,
    int n_sites,
    int n_subsample_pairs
);


// ============================================================
// Convergence Check
// ============================================================
// Max-reduction of |messages - messages_prev| across all entries.

/**
 * @brief Check convergence by computing max absolute difference.
 *
 * @param messages       current messages [total_elements]
 * @param messages_prev  previous iteration messages [total_elements]
 * @param total_elements number of float elements to compare
 * @param max_delta_out  output: max |messages - messages_prev| (device pointer, single float)
 */
void check_convergence_gpu(
    const float* messages,
    const float* messages_prev,
    int total_elements,
    float* max_delta_out
);
