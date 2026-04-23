#pragma once

#include "types.h"
#include <cstdint>

// Bitpack a genotype matrix G[n][S] (uint8, row-major) into packed[n][n_words] (uint64)
void bitpack_genotypes_gpu(const uint8_t* G, uint64_t* packed,
                           int n, int S, int n_words);

// Unpack packed[n][n_words] back to G[n][S] (for validation)
void unpack_genotypes_gpu(const uint64_t* packed, uint8_t* G,
                          int n, int S, int n_words);

// Compute pairwise XOR prefix scan for a batch of pairs
// prefix_out[n_pairs][S]: cumulative difference count at each site
void pairwise_prefix_scan_gpu(const uint64_t* packed, int n_words, int S,
                              const int* pair_i, const int* pair_j,
                              int n_pairs, int64_t* prefix_out);

// Compute windowed divergence from prefix scans
// div_out[n_pairs][S]: windowed difference count at each site
void windowed_divergence_gpu(const int64_t* prefix, int S,
                             int n_pairs, int window_sites,
                             float* div_out);
