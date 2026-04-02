#pragma once

#include "types.h"
#include <cstdint>

// Gamma-SMC forward filtering on GPU.
//
// Maintains Gamma(alpha, beta) posterior over TMRCA at each site using
// moment-matched recombination transitions and conjugate Poisson emissions.
// O(1) state per pair (vs O(K) for discrete HMM).
//
// Output layout is site-major [out_S × n_pairs] for coalesced GPU writes.
// lower_out / upper_out may be NULL to skip CI computation (mean-only mode).
void gamma_smc_forward_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float mu, float rho, float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    float* tmrca_mean_out,   // [out_S × n_pairs], required
    float* tmrca_lower_out,  // [out_S × n_pairs] or NULL
    float* tmrca_upper_out,  // [out_S × n_pairs] or NULL
    int stride = 1);

// Quantized output: log-scale uint8 or uint4 (packed).
// q_out layout:
//   bits=8: [out_S × n_pairs] uint8
//   bits=4: [ceil(out_S/2) × n_pairs] uint8 (two 4-bit values per byte)
// Dequantize: t = exp(log_min + (q / (2^bits - 1)) * (log_max - log_min))
void gamma_smc_forward_quantized_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float mu, float rho, float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    unsigned char* q_out,
    float log_min, float log_max,
    int stride = 1, int bits = 8);
