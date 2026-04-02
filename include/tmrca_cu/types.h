#pragma once

#include <cstdint>
#include <cmath>

// Maximum number of discrete time bins (K=32/64/128 supported)
#define TMRCA_K_MAX 128

// Pair tile and site block sizes (tuned for single A100)
#define PAIR_TILE_SIZE 2048
#define SITE_BLOCK_SIZE 65536

// === Core genotype data ===
typedef struct {
    uint64_t* packed;       // Bitpacked genotype matrix [n × n_words]
    double* positions;      // Physical positions of segregating sites [S]
    int n;                  // Number of haplotypes
    int S;                  // Number of segregating sites
    int n_words;            // Number of uint64 words per haplotype = ceil(S / 64)
} GenotypeMatrix;

// === Recombination and mutation maps ===
typedef struct {
    double* mu;             // Per-site mutation rate [S]
    double* rho;            // Per-site recombination rate [S]
    double* cum_rho;        // Cumulative recombination distance [S]
} GeneticMaps;

// === Demographic model ===
typedef struct {
    double* Ne;             // Piecewise-constant Ne values [M]
    double* epoch_boundaries; // Time boundaries for Ne epochs [M+1]
    int M;                  // Number of epochs
    double* coal_prior;     // Coalescent prior q[k] for each time bin [K]
    double* cum_coal_rate;  // Cumulative coalescent rate integral [K]
} Demography;

// === HMM parameters (precomputed, read-only) ===
typedef struct {
    double time_midpoints[TMRCA_K_MAX];    // Midpoint of each time bin
    double time_boundaries[TMRCA_K_MAX+1]; // Boundaries of time bins
    double coal_prior[TMRCA_K_MAX];        // q[k] = prior probability of coalescing in bin k
} HMMParams;

// === Output ===
typedef struct {
    float* tmrca_mean;      // E[T_ij(s) | data], shape [n_pairs × S]
    float* tmrca_lower;     // 2.5th percentile
    float* tmrca_upper;     // 97.5th percentile
    float* pi;              // Per-site mean pairwise divergence [S]
    float* tmrca_site_mean; // Mean TMRCA across pairs [S]
    float* tmrca_site_var;  // Variance of TMRCA across pairs [S]
} Output;

// Helper: map linear pair index p -> (i, j) where i > j
inline void pair_to_ij(long long p, int* i, int* j) {
    // i = floor((1 + sqrt(1 + 8p)) / 2)
    double r = (1.0 + sqrt(1.0 + 8.0 * (double)p)) / 2.0;
    *i = (int)r;
    if ((long long)(*i) * ((*i) - 1) / 2 > p) (*i)--;
    *j = (int)(p - (long long)(*i) * ((*i) - 1) / 2);
}

// Helper: map (i, j) -> linear pair index (i > j)
inline long long ij_to_pair(int i, int j) {
    if (i < j) { int t = i; i = j; j = t; }
    return (long long)i * (i - 1) / 2 + j;
}
