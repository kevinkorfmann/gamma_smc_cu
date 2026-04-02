#pragma once

#include "types.h"

// Output mode for HMM forward-backward
enum HMMOutputMode {
    FULL_GAMMA = 0,      // Write full gamma[n_pairs][S][K] array
    SUMMARY_ONLY = 1,    // Fused: compute mean/CI on the fly, skip gamma D2H
    SUMMARY_AND_EM = 2   // Fused: summaries + atomicAdd q accumulation for EM
};

// Compute coalescent prior q[k] under constant Ne
// time_boundaries[K+1], coal_prior_out[K]
void compute_coalescent_prior(double Ne, double t_max,
                              double* time_boundaries,
                              double* time_midpoints,
                              double* coal_prior_out,
                              int K = 32);

// Compute SFS from bitpacked genotype matrix
// sfs_out[n+1]: allele frequency spectrum
void compute_sfs_gpu(const uint64_t* packed, int n, int S, int n_words,
                     int* sfs_out);

// Run batched HMM forward-backward for a set of pairs
//
// gamma_out[n_pairs][S][K]: posterior marginals (float)
//   - Must be allocated even in SUMMARY_ONLY/SUMMARY_AND_EM modes (used as
//     alpha scratch during backward pass), but contents are NOT meaningful
//     after the call in those modes.
// log_lik_out[n_pairs]: total log-likelihood per pair
// tmrca_mean_out, tmrca_lower_out, tmrca_upper_out: [n_pairs × S] summaries
//   - Only written in SUMMARY_ONLY / SUMMARY_AND_EM modes. May be NULL in FULL_GAMMA mode.
// q_accum_out[K]: atomicAdd accumulator for EM
//   - Only written in SUMMARY_AND_EM mode. May be NULL otherwise.
//   - Caller must zero this buffer before the call.
// K must be 32, 64, or 128
void hmm_forward_backward_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    const double* mu, const double* cum_rho,
    const double* time_midpoints, const double* coal_prior,
    const float* messages,   // [n_pairs × S × K] or NULL for prior-only
    const int* pair_i, const int* pair_j, int n_pairs,
    float* gamma_out, double* log_lik_out,
    float* tmrca_mean_out, float* tmrca_lower_out, float* tmrca_upper_out,
    double* q_accum_out,
    int K = 32, HMMOutputMode mode = FULL_GAMMA);

// Extract posterior summaries from gamma (kept for backward compat)
void extract_summaries_gpu(const float* gamma, int n_pairs, int S,
                           const double* time_midpoints,
                           float* tmrca_mean, float* tmrca_lower,
                           float* tmrca_upper,
                           int K = 32);

// Aggregate posteriors across pairs and sites: mean gamma per bin
void aggregate_posteriors_gpu(const float* gamma, int n_pairs, int S, int K,
                              double* q_empirical_out);

// Blend empirical prior with old prior: q_new = (1-alpha)*q_old + alpha*q_emp, then normalize
void blend_prior_gpu(const double* q_old, const double* q_emp, double alpha,
                     int K, double* q_new_out);
