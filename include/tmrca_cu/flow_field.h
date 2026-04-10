#pragma once

#include <cstdint>

// Flow field grid dimensions (from Schweiger's default_flow_field.txt)
constexpr int FF_MEAN_N = 51;   // log10(mean) grid: [-5, 2]
constexpr int FF_CV_N   = 50;   // log10(cv) grid:   [-2, 0]
constexpr int FF_GRID   = FF_MEAN_N * FF_CV_N;  // 2550

// Base flow field: (u, v) displacement rates in (log10_mean, log10_cv) space.
struct FlowFieldData {
    float u[FF_GRID];  // [51 × 50], row-major (mean-major)
    float v[FF_GRID];
    float mean_log10_min;   // -5.0
    float mean_log10_max;   //  2.0
    float cv_log10_min;     // -2.0
    float cv_log10_max;     //  0.0
};

// Load base flow field from Schweiger's text file.
bool load_flow_field(const char* path, FlowFieldData& out);

// Multi-step cache: precomputed flow field results for 1..n_max_steps.
// For each step count n and grid point (row, col), stores the (mean_log10, cv_log10)
// after applying n iterations of: mutation emission → flow field recombination.
// Layout: mean[n * FF_GRID + row * FF_CV_N + col], same for cv.
struct FlowFieldCache {
    int n_max_steps;
    float* missing_mean;  // [n_max_steps × FF_MEAN_N × FF_CV_N]
    float* missing_cv;    // [n_max_steps × FF_MEAN_N × FF_CV_N]
    float* mean;  // [n_max_steps × FF_MEAN_N × FF_CV_N]
    float* cv;    // [n_max_steps × FF_MEAN_N × FF_CV_N]
    float* fwd_hom_site_mean;
    float* fwd_hom_site_cv;
    float* fwd_het_site_mean;
    float* fwd_het_site_cv;
    float* bwd_hom_site_mean;
    float* bwd_hom_site_cv;
    float* bwd_het_site_mean;
    float* bwd_het_site_cv;
};

struct FlowFieldDeviceCacheView {
    const float* missing_mean;
    const float* missing_cv;
    const float* mean;
    const float* cv;
    const float* fwd_hom_site_mean;
    const float* fwd_hom_site_cv;
    const float* fwd_het_site_mean;
    const float* fwd_het_site_cv;
    const float* bwd_hom_site_mean;
    const float* bwd_hom_site_cv;
    const float* bwd_het_site_mean;
    const float* bwd_het_site_cv;
    int n_max_steps;
};

// Build multi-step "hom" cache on CPU.
// scaled_rho = 4*Ne*rho (displacement per bp per flow field step)
// scaled_mu  = 4*Ne*mu  (mutation emission per bp in scaled coordinates)
// Caller owns the returned arrays (allocated with new[]).
FlowFieldCache build_flow_field_cache(
    const FlowFieldData& ff,
    int n_max_steps,
    float scaled_rho,
    float scaled_mu);

void free_flow_field_cache(FlowFieldCache& cache);

// GPU forward-backward with flow field transition model (iterative, no cache).
void gamma_smc_flow_fb_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float mu, float rho, float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const float* d_flow_u, const float* d_flow_v,
    float* fwd_buf,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out);

// GPU forward-backward with precomputed multi-step cache (fast path).
// d_cache_mean, d_cache_cv: [n_max_steps × FF_GRID] on GPU.
// At each site, gap→single bilinear lookup. No transcendentals in the hot path.
//
// posterior_alpha_out / posterior_beta_out are optional; pass nullptr to skip.
// When non-null, they receive the per-site combined Gamma posterior parameters
// (a_s, b_s) in scaled coalescent time (T_scaled = T / (2*Ne)). The kernel
// branches on the pointer uniformly across the warp, so the cost when unused
// is negligible.
void gamma_smc_flow_cached_fb_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    FlowFieldDeviceCacheView cache,
    float* fwd_buf,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out,
    float* posterior_alpha_out = nullptr,
    float* posterior_beta_out = nullptr);

// GPU forward-backward on a padded site block [site_start, site_start + block_S).
// Output arrays are site-major over the local block shape [block_S × n_pairs].
// posterior_alpha_out / posterior_beta_out follow the same convention as above.
void gamma_smc_flow_cached_fb_block_gpu(
    const uint64_t* packed, int n_words,
    const double* positions,
    int site_start, int block_S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    FlowFieldDeviceCacheView cache,
    float* fwd_buf,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out,
    float* posterior_alpha_out = nullptr,
    float* posterior_beta_out = nullptr);

// GPU forward-only with interleaved float2 cache. No forward buffer needed.
// Single pass: outputs mean (and optionally CI) directly.
// d_cache: [n_max_steps × FF_GRID] interleaved (mean, cv) float pairs on GPU.
void gamma_smc_flow_cached_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache,          // float2* on GPU: [n_max_steps × FF_GRID]
    int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,       // nullptr for mean-only
    float* tmrca_upper_out);      // nullptr for mean-only

// Fused flow FB + per-site reduction. Returns [S] floats instead of [S × n_pairs].
void gamma_smc_flow_cached_fb_reduce_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    FlowFieldDeviceCacheView cache,
    float* fwd_buf,
    float* site_mean_out,
    float* site_min_out,
    float* site_max_out);

// Debug helper: run the same cached forward kernel used by cached FB and
// return the raw forward state buffer [mean_log10, cv_log10].
void gamma_smc_flow_cached_forward_states_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    const int* pair_i, const int* pair_j, int n_pairs,
    FlowFieldDeviceCacheView cache,
    float* fwd_buf);
