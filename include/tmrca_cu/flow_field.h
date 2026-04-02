#pragma once

#include <cstdint>

// Flow field grid dimensions (from Schweiger's default_flow_field.txt)
constexpr int FF_MEAN_N = 51;   // log10(mean) grid: [-5, 2]
constexpr int FF_CV_N   = 50;   // log10(cv) grid:   [-2, 0]

// Base flow field: (u, v) displacement rates in (log10_mean, log10_cv) space.
// Update rule: mean_log10 += u * scaled_rho, cv_log10 += v * scaled_rho
// where scaled_rho = 2 * Ne * rho * gap_bp.
struct FlowFieldData {
    float u[FF_MEAN_N * FF_CV_N];  // [51 × 50], row-major (mean-major)
    float v[FF_MEAN_N * FF_CV_N];
    float mean_log10_min;   // -5.0
    float mean_log10_max;   //  2.0
    float cv_log10_min;     // -2.0
    float cv_log10_max;     //  0.0
};

// Load base flow field from Schweiger's text file.
// Returns false on parse error.
bool load_flow_field(const char* path, FlowFieldData& out);

// GPU forward-backward with flow field transition model.
//
// Operates in Schweiger's (log10_mean, log10_cv) coordinate system using
// the precomputed flow field for exact recombination transitions (no
// moment-matching). Performs two passes:
//   Forward:  left→right, stores (mean_log10, cv_log10) at each site
//   Backward: right→left, combines with forward via:
//     α_smooth = α_f + α_b − 1,  β_smooth = β_f + β_b − 1  (scaled prior β=1)
//
// Output layout: site-major [S × n_pairs].
// lower_out / upper_out may be NULL to skip CI (mean-only mode).
void gamma_smc_flow_fb_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float mu, float rho, float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const float* d_flow_u, const float* d_flow_v,  // [FF_MEAN_N * FF_CV_N] on GPU
    float* fwd_buf,         // scratch: [2 * S * n_pairs] for forward (mean, cv)
    float* tmrca_mean_out,  // [S × n_pairs]
    float* tmrca_lower_out, // [S × n_pairs] or NULL
    float* tmrca_upper_out);// [S × n_pairs] or NULL
