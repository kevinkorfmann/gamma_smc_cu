#include "tmrca_cu/flow_field.h"
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>

// ============================================================
// Constants
// ============================================================
// Grid: mean_log10 [-5, 2] × cv_log10 [-2, 0]
static constexpr float MEAN_LOG10_MIN  = -5.0f;
static constexpr float MEAN_LOG10_MAX  =  2.0f;
static constexpr float CV_LOG10_MIN    = -2.0f;
static constexpr float CV_LOG10_MAX    =  0.0f;
static constexpr float MEAN_STEP = (MEAN_LOG10_MAX - MEAN_LOG10_MIN) / (FF_MEAN_N - 1);
static constexpr float CV_STEP   = (CV_LOG10_MAX - CV_LOG10_MIN) / (FF_CV_N - 1);
static constexpr float MEAN_STEP_INV = 1.0f / MEAN_STEP;
static constexpr float CV_STEP_INV   = 1.0f / CV_STEP;

// Max scaled recombination per sub-step.
// Larger = fewer iterations = faster but less accurate.
// 0.1 gives good accuracy for typical human genetics params.
static constexpr float MAX_STEP_RHO = 0.1f;

// ============================================================
// Shared memory layout: [FF_MEAN_N * FF_CV_N] U then V = 2*2550 = 5100 floats = 20.4 KB
// ============================================================
static constexpr int FF_SIZE = FF_MEAN_N * FF_CV_N;  // 2550

// ============================================================
// Device helpers
// ============================================================

// Bilinear interpolation from shared memory flow field table.
__device__ __forceinline__ float bilinear_smem(
    const float* __restrict__ table,  // shared memory [FF_SIZE]
    float mean_log10, float cv_log10)
{
    float fm = (mean_log10 - MEAN_LOG10_MIN) * MEAN_STEP_INV;
    float fc = (cv_log10 - CV_LOG10_MIN) * CV_STEP_INV;

    fm = fmaxf(0.0f, fminf(fm, (float)(FF_MEAN_N - 2)));
    fc = fmaxf(0.0f, fminf(fc, (float)(FF_CV_N - 2)));

    int m0 = (int)fm;
    int c0 = (int)fc;
    float wm = fm - (float)m0;
    float wc = fc - (float)c0;

    int base = m0 * FF_CV_N + c0;
    float v00 = table[base];
    float v01 = table[base + 1];
    float v10 = table[base + FF_CV_N];
    float v11 = table[base + FF_CV_N + 1];

    return __fmaf_rn(wm, __fmaf_rn(wc, v11, (1.0f - wc) * v10),
           (1.0f - wm) * __fmaf_rn(wc, v01, (1.0f - wc) * v00));
}

// Apply flow field for a gap: mutation emission + recombination transition.
// Uses adaptive sub-stepping to keep per-step displacement bounded.
__device__ __forceinline__ void flow_field_advance(
    float& mean_log10, float& cv_log10,
    const float* __restrict__ smem_u,
    const float* __restrict__ smem_v,
    float scaled_rho_total,
    float scaled_mu_total)
{
    if (scaled_rho_total < 1e-12f) return;

    int n_iter = max(1, (int)ceilf(scaled_rho_total * (1.0f / MAX_STEP_RHO)));
    float inv_n = __frcp_rn((float)n_iter);
    float rho_per = scaled_rho_total * inv_n;
    float mu_per  = scaled_mu_total * inv_n;

    for (int i = 0; i < n_iter; i++) {
        // 1. Mutation emission: beta += mu_per (in scaled coords)
        float a_log = -2.0f * cv_log10;
        float b_log = a_log - mean_log10;
        float b_lin = __exp10f(b_log) + mu_per;
        b_log = __log10f(b_lin);
        mean_log10 = a_log - b_log;

        // 2. Recombination via flow field
        float u = bilinear_smem(smem_u, mean_log10, cv_log10);
        float v = bilinear_smem(smem_v, mean_log10, cv_log10);
        mean_log10 += u * rho_per;
        cv_log10   += v * rho_per;

        // Clamp
        mean_log10 = fmaxf(MEAN_LOG10_MIN, fminf(mean_log10, MEAN_LOG10_MAX));
        cv_log10   = fmaxf(CV_LOG10_MIN, fminf(cv_log10, CV_LOG10_MAX));
    }
}

// (mean_log10, cv_log10) → (alpha, beta) in linear space
__device__ __forceinline__ void mc_to_ab(
    float m, float c, float& alpha, float& beta)
{
    float a_log = -2.0f * c;
    alpha = __exp10f(a_log);
    beta  = __exp10f(a_log - m);
}

// (alpha, beta) → (mean_log10, cv_log10)
__device__ __forceinline__ void ab_to_mc(
    float alpha, float beta, float& m, float& c)
{
    float a_log = __log10f(fmaxf(alpha, 1e-30f));
    m = a_log - __log10f(fmaxf(beta, 1e-30f));
    c = -0.5f * a_log;
}

// ============================================================
// Forward pass kernel — loads flow field into shared memory
// ============================================================
__global__ void gamma_smc_flow_forward_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float scaled_rho_per_bp,
    float scaled_mu_per_bp,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float* __restrict__ flow_u,  // global [FF_SIZE]
    const float* __restrict__ flow_v,  // global [FF_SIZE]
    float* __restrict__ fwd_mean,      // [S × n_pairs]
    float* __restrict__ fwd_cv)        // [S × n_pairs]
{
    // Load flow field into shared memory (cooperative load)
    __shared__ float smem_u[FF_SIZE];
    __shared__ float smem_v[FF_SIZE];
    for (int i = threadIdx.x; i < FF_SIZE; i += blockDim.x) {
        smem_u[i] = flow_u[i];
        smem_v[i] = flow_v[i];
    }
    __syncthreads();

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float mean_log10 = 0.0f;
    float cv_log10   = 0.0f;

    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        float gap = (float)(pos - prev_pos);
        prev_pos = pos;

        if (s > 0 && gap > 0.0f) {
            flow_field_advance(mean_log10, cv_log10, smem_u, smem_v,
                               scaled_rho_per_bp * gap,
                               scaled_mu_per_bp * gap);
        }

        // Site emission
        int w = s >> 6;
        int bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha, beta;
            mc_to_ab(mean_log10, cv_log10, alpha, beta);
            alpha += 1.0f;
            ab_to_mc(alpha, beta, mean_log10, cv_log10);
        }

        long long idx = (long long)s * n_pairs + pid;
        fwd_mean[idx] = mean_log10;
        fwd_cv[idx]   = cv_log10;
    }
}

// ============================================================
// Backward + combine kernel
// ============================================================
template<bool WRITE_CI>
__global__ void gamma_smc_flow_backward_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float scaled_rho_per_bp,
    float scaled_mu_per_bp,
    float Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float* __restrict__ flow_u,
    const float* __restrict__ flow_v,
    const float* __restrict__ fwd_mean,
    const float* __restrict__ fwd_cv,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    __shared__ float smem_u[FF_SIZE];
    __shared__ float smem_v[FF_SIZE];
    for (int i = threadIdx.x; i < FF_SIZE; i += blockDim.x) {
        smem_u[i] = flow_u[i];
        smem_v[i] = flow_v[i];
    }
    __syncthreads();

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float mean_log10 = 0.0f;
    float cv_log10   = 0.0f;
    float unscale = 2.0f * Ne;

    int cur_word = -1;
    uint64_t xor_w = 0;

    for (int s = S - 1; s >= 0; s--) {
        // Backward state BEFORE emission at s
        float bwd_a, bwd_b;
        mc_to_ab(mean_log10, cv_log10, bwd_a, bwd_b);

        // Forward state
        long long idx = (long long)s * n_pairs + pid;
        float fwd_a, fwd_b;
        mc_to_ab(fwd_mean[idx], fwd_cv[idx], fwd_a, fwd_b);

        // Combine (scaled prior α=1, β=1)
        float a_s = fmaxf(fwd_a + bwd_a - 1.0f, 1.0f);
        float b_s = fmaxf(fwd_b + bwd_b - 1.0f, 1e-10f);
        float mean_gen = (a_s / b_s) * unscale;
        mean_out[idx] = mean_gen;

        if constexpr (WRITE_CI) {
            float inv9a = __frcp_rn(9.0f * a_s);
            float sq = __fsqrt_rn(inv9a);
            float base = 1.0f - inv9a;
            float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
            float hi_f = base + 1.96f * sq;
            lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
            upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
        }

        // Absorb emission
        int w = s >> 6;
        int bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha, beta;
            mc_to_ab(mean_log10, cv_log10, alpha, beta);
            alpha += 1.0f;
            ab_to_mc(alpha, beta, mean_log10, cv_log10);
        }

        // Transition to s-1
        if (s > 0) {
            float gap = (float)(positions[s] - positions[s - 1]);
            if (gap > 0.0f) {
                flow_field_advance(mean_log10, cv_log10, smem_u, smem_v,
                                   scaled_rho_per_bp * gap,
                                   scaled_mu_per_bp * gap);
            }
        }
    }
}

// ============================================================
// Host launcher
// ============================================================
void gamma_smc_flow_fb_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float mu, float rho, float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const float* d_flow_u, const float* d_flow_v,
    float* fwd_buf,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    float lambda = 1.0f / (2.0f * Ne);
    float scaled_rho_per_bp = 2.0f * Ne * rho;
    float scaled_mu_per_bp  = 2.0f * mu / lambda;  // = 4*Ne*mu

    const int block = 256;
    int grid = (n_pairs + block - 1) / block;

    float* fwd_mean = fwd_buf;
    float* fwd_cv   = fwd_buf + (long long)S * n_pairs;

    // Forward pass
    gamma_smc_flow_forward_kernel<<<grid, block>>>(
        packed, n_words, positions, S,
        scaled_rho_per_bp, scaled_mu_per_bp,
        pair_i, pair_j, n_pairs,
        d_flow_u, d_flow_v,
        fwd_mean, fwd_cv);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_flow forward: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    // Backward + combine
    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_flow_backward_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S,
            scaled_rho_per_bp, scaled_mu_per_bp, Ne,
            pair_i, pair_j, n_pairs,
            d_flow_u, d_flow_v,
            fwd_mean, fwd_cv,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_flow_backward_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S,
            scaled_rho_per_bp, scaled_mu_per_bp, Ne,
            pair_i, pair_j, n_pairs,
            d_flow_u, d_flow_v,
            fwd_mean, fwd_cv,
            tmrca_mean_out, nullptr, nullptr);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_flow backward: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// CACHED VERSION: Multi-step cache eliminates per-site iteration.
// Each gap → single bilinear lookup via __ldg() (read-only texture cache).
// No shared memory loading per site; L2 cache handles reuse across blocks.
// ============================================================

// Bilinear interpolation directly from global memory via __ldg().
// ptr points to one step's grid: cache + step_idx * FF_GRID.
__device__ __forceinline__ float cache_bilinear_ldg(
    const float* __restrict__ ptr,
    float mean_log10, float cv_log10)
{
    float fm = (mean_log10 - MEAN_LOG10_MIN) * MEAN_STEP_INV;
    float fc = (cv_log10 - CV_LOG10_MIN) * CV_STEP_INV;
    fm = fmaxf(0.0f, fminf(fm, (float)(FF_MEAN_N - 2)));
    fc = fmaxf(0.0f, fminf(fc, (float)(FF_CV_N - 2)));

    int m0 = (int)fm, c0 = (int)fc;
    float wm = fm - (float)m0, wc = fc - (float)c0;

    int base = m0 * FF_CV_N + c0;
    float v00 = __ldg(ptr + base);
    float v01 = __ldg(ptr + base + 1);
    float v10 = __ldg(ptr + base + FF_CV_N);
    float v11 = __ldg(ptr + base + FF_CV_N + 1);

    return __fmaf_rn(wm, __fmaf_rn(wc, v11, (1.0f - wc) * v10),
           (1.0f - wm) * __fmaf_rn(wc, v01, (1.0f - wc) * v00));
}

// Apply cache lookup for `gap_steps` bp.
// Decomposes into chunks of n_max_steps if needed.
__device__ __forceinline__ void cache_advance(
    float& m, float& c,
    const float* __restrict__ cache_mean,
    const float* __restrict__ cache_cv,
    int gap_steps, int n_max_steps)
{
    while (gap_steps > n_max_steps) {
        const float* pm = cache_mean + (size_t)(n_max_steps - 1) * FF_GRID;
        const float* pc = cache_cv   + (size_t)(n_max_steps - 1) * FF_GRID;
        float m_new = cache_bilinear_ldg(pm, m, c);
        float c_new = cache_bilinear_ldg(pc, m, c);
        m = m_new; c = c_new;
        gap_steps -= n_max_steps;
    }
    if (gap_steps > 0) {
        const float* pm = cache_mean + (size_t)(gap_steps - 1) * FF_GRID;
        const float* pc = cache_cv   + (size_t)(gap_steps - 1) * FF_GRID;
        float m_new = cache_bilinear_ldg(pm, m, c);
        float c_new = cache_bilinear_ldg(pc, m, c);
        m = m_new; c = c_new;
    }
}

// ============================================================
// Interleaved float2 cache: (mean, cv) packed per grid point.
// Halves L2 requests: 4 float2 reads vs 8 float reads.
// ============================================================
__device__ __forceinline__ void cache_bilinear_f2(
    const float2* __restrict__ ptr,  // one step's grid: [FF_GRID]
    float mean_log10, float cv_log10,
    float& m_out, float& c_out)
{
    float fm = (mean_log10 - MEAN_LOG10_MIN) * MEAN_STEP_INV;
    float fc = (cv_log10 - CV_LOG10_MIN) * CV_STEP_INV;
    fm = fmaxf(0.0f, fminf(fm, (float)(FF_MEAN_N - 2)));
    fc = fmaxf(0.0f, fminf(fc, (float)(FF_CV_N - 2)));

    int m0 = (int)fm, c0 = (int)fc;
    float wm = fm - (float)m0, wc = fc - (float)c0;

    int base = m0 * FF_CV_N + c0;
    float2 v00 = __ldg(ptr + base);
    float2 v01 = __ldg(ptr + base + 1);
    float2 v10 = __ldg(ptr + base + FF_CV_N);
    float2 v11 = __ldg(ptr + base + FF_CV_N + 1);

    float omwm = 1.0f - wm, omwc = 1.0f - wc;
    m_out = __fmaf_rn(wm, __fmaf_rn(wc, v11.x, omwc * v10.x),
            omwm * __fmaf_rn(wc, v01.x, omwc * v00.x));
    c_out = __fmaf_rn(wm, __fmaf_rn(wc, v11.y, omwc * v10.y),
            omwm * __fmaf_rn(wc, v01.y, omwc * v00.y));
}

__device__ __forceinline__ void cache_advance_f2(
    float& m, float& c,
    const float2* __restrict__ cache,  // [n_max_steps × FF_GRID]
    int gap_steps, int n_max_steps)
{
    while (gap_steps > n_max_steps) {
        const float2* p = cache + (size_t)(n_max_steps - 1) * FF_GRID;
        cache_bilinear_f2(p, m, c, m, c);
        gap_steps -= n_max_steps;
    }
    if (gap_steps > 0) {
        const float2* p = cache + (size_t)(gap_steps - 1) * FF_GRID;
        cache_bilinear_f2(p, m, c, m, c);
    }
}

// ============================================================
// Forward pass with cache
// ============================================================
__global__ void gamma_smc_cached_forward_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float* __restrict__ cache_mean,
    const float* __restrict__ cache_cv,
    int n_max_steps,
    float* __restrict__ fwd_mean,
    float* __restrict__ fwd_cv_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        int gap_steps = (int)(pos - prev_pos + 0.5);
        prev_pos = pos;

        if (s > 0 && gap_steps > 0) {
            cache_advance(m, c, cache_mean, cache_cv, gap_steps, n_max_steps);
        }

        // Site emission: het → alpha += 1
        int w = s >> 6;
        int bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        long long idx = (long long)s * n_pairs + pid;
        fwd_mean[idx] = m;
        fwd_cv_out[idx] = c;
    }
}

// ============================================================
// Backward pass + combine with cache
// ============================================================
template<bool WRITE_CI>
__global__ void gamma_smc_cached_backward_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float* __restrict__ cache_mean,
    const float* __restrict__ cache_cv,
    int n_max_steps,
    const float* __restrict__ fwd_mean_in,
    const float* __restrict__ fwd_cv_in,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    float unscale = 2.0f * Ne;
    int cur_word = -1;
    uint64_t xor_w = 0;

    for (int s = S - 1; s >= 0; s--) {
        // Backward state BEFORE emission
        float bwd_a = __exp10f(-2.0f * c);
        float bwd_b = __exp10f(-2.0f * c - m);

        // Forward state
        long long idx = (long long)s * n_pairs + pid;
        float fm = fwd_mean_in[idx], fc = fwd_cv_in[idx];
        float fwd_a = __exp10f(-2.0f * fc);
        float fwd_b = __exp10f(-2.0f * fc - fm);

        // Combine
        float a_s = fmaxf(fwd_a + bwd_a - 1.0f, 1.0f);
        float b_s = fmaxf(fwd_b + bwd_b - 1.0f, 1e-10f);
        float mean_gen = (a_s / b_s) * unscale;
        mean_out[idx] = mean_gen;

        if constexpr (WRITE_CI) {
            float inv9a = __frcp_rn(9.0f * a_s);
            float sq = __fsqrt_rn(inv9a);
            float base = 1.0f - inv9a;
            float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
            float hi_f = base + 1.96f * sq;
            lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
            upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
        }

        // Absorb emission
        int w = s >> 6;
        int bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        // Transition to s-1
        if (s > 0) {
            int gap_steps = (int)(positions[s] - positions[s - 1] + 0.5);
            if (gap_steps > 0)
                cache_advance(m, c, cache_mean, cache_cv, gap_steps, n_max_steps);
        }
    }
}

// ============================================================
// Host launcher — cached version
// ============================================================
void gamma_smc_flow_cached_fb_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const float* d_cache_mean, const float* d_cache_cv,
    int n_max_steps,
    float* fwd_buf,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;

    float* fwd_mean = fwd_buf;
    float* fwd_cv   = fwd_buf + (long long)S * n_pairs;

    // Forward
    gamma_smc_cached_forward_kernel<<<grid, block>>>(
        packed, n_words, positions, S,
        pair_i, pair_j, n_pairs,
        d_cache_mean, d_cache_cv, n_max_steps,
        fwd_mean, fwd_cv);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_cached forward: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    // Backward + combine
    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_cached_backward_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S, Ne,
            pair_i, pair_j, n_pairs,
            d_cache_mean, d_cache_cv, n_max_steps,
            fwd_mean, fwd_cv,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_cached_backward_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S, Ne,
            pair_i, pair_j, n_pairs,
            d_cache_mean, d_cache_cv, n_max_steps,
            fwd_mean, fwd_cv,
            tmrca_mean_out, nullptr, nullptr);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_cached backward: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// TEXTURE-BASED forward-only kernel: hardware bilinear interpolation
// replaces ~26 instructions of software bilinear with 1 tex fetch.
// ============================================================

// Advance (m, c) using hardware-interpolated layered 2D texture.
// tex: float2 layered texture [n_layers × FF_MEAN_N × FF_CV_N]
//   x = cv dim (width = FF_CV_N), y = mean dim (height = FF_MEAN_N)
//   layer = step index (0-based)
__device__ __forceinline__ void cache_advance_tex(
    float& m, float& c,
    cudaTextureObject_t tex,
    int gap_steps, int n_tex_layers)
{
    while (gap_steps > n_tex_layers) {
        float fm = (m - MEAN_LOG10_MIN) * MEAN_STEP_INV;
        float fc = (c - CV_LOG10_MIN) * CV_STEP_INV;
        float2 mc = tex2DLayered<float2>(tex, fc + 0.5f, fm + 0.5f, n_tex_layers - 1);
        m = mc.x; c = mc.y;
        gap_steps -= n_tex_layers;
    }
    if (gap_steps > 0) {
        float fm = (m - MEAN_LOG10_MIN) * MEAN_STEP_INV;
        float fc = (c - CV_LOG10_MIN) * CV_STEP_INV;
        float2 mc = tex2DLayered<float2>(tex, fc + 0.5f, fm + 0.5f, gap_steps - 1);
        m = mc.x; c = mc.y;
    }
}

template<bool WRITE_CI>
__global__ __launch_bounds__(256, 4)
void gamma_smc_tex_fwd_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float two_Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    cudaTextureObject_t cache_tex,
    int n_tex_layers,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        int gap_steps = (int)(pos - prev_pos + 0.5);
        prev_pos = pos;

        if (s > 0 && gap_steps > 0)
            cache_advance_tex(m, c, cache_tex, gap_steps, n_tex_layers);

        // Het emission (rare: ~1% of sites)
        int w = s >> 6, bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        // Output: mean_gen = 10^m * 2Ne
        long long idx = (long long)s * n_pairs + pid;
        float mean_gen = exp2f(m * 3.321928094887362f) * two_Ne;
        mean_out[idx] = mean_gen;

        if constexpr (WRITE_CI) {
            float alpha = __exp10f(-2.0f * c);
            float inv9a = __frcp_rn(9.0f * alpha);
            float sq = __fsqrt_rn(inv9a);
            float base = 1.0f - inv9a;
            float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
            float hi_f = base + 1.96f * sq;
            lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
            upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
        }
    }
}

// Host launcher — texture forward-only
void gamma_smc_flow_tex_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    cudaTextureObject_t cache_tex, int n_tex_layers,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;
    float two_Ne = 2.0f * Ne;

    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_tex_fwd_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            cache_tex, n_tex_layers,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_tex_fwd_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            cache_tex, n_tex_layers,
            tmrca_mean_out, nullptr, nullptr);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_tex fwd: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// HALF-PRECISION (fp16) cache: halves L2 traffic per bilinear lookup.
// Reads __half2 (4B) instead of float2 (8B), converts to fp32 for arithmetic.
// ============================================================
__device__ __forceinline__ void cache_bilinear_h2(
    const __half2* __restrict__ ptr,  // one step's grid: [FF_GRID]
    float mean_log10, float cv_log10,
    float& m_out, float& c_out)
{
    float fm = (mean_log10 - MEAN_LOG10_MIN) * MEAN_STEP_INV;
    float fc = (cv_log10 - CV_LOG10_MIN) * CV_STEP_INV;
    fm = fmaxf(0.0f, fminf(fm, (float)(FF_MEAN_N - 2)));
    fc = fmaxf(0.0f, fminf(fc, (float)(FF_CV_N - 2)));

    int m0 = (int)fm, c0 = (int)fc;
    float wm = fm - (float)m0, wc = fc - (float)c0;

    int base = m0 * FF_CV_N + c0;
    // Each __ldg reads 4B (half2) instead of 8B (float2)
    __half2 h00 = __ldg(ptr + base);
    __half2 h01 = __ldg(ptr + base + 1);
    __half2 h10 = __ldg(ptr + base + FF_CV_N);
    __half2 h11 = __ldg(ptr + base + FF_CV_N + 1);

    // Convert to fp32 for interpolation
    float2 v00 = __half22float2(h00);
    float2 v01 = __half22float2(h01);
    float2 v10 = __half22float2(h10);
    float2 v11 = __half22float2(h11);

    float omwm = 1.0f - wm, omwc = 1.0f - wc;
    m_out = __fmaf_rn(wm, __fmaf_rn(wc, v11.x, omwc * v10.x),
            omwm * __fmaf_rn(wc, v01.x, omwc * v00.x));
    c_out = __fmaf_rn(wm, __fmaf_rn(wc, v11.y, omwc * v10.y),
            omwm * __fmaf_rn(wc, v01.y, omwc * v00.y));
}

__device__ __forceinline__ void cache_advance_h2(
    float& m, float& c,
    const __half2* __restrict__ cache,
    int gap_steps, int n_max_steps)
{
    while (gap_steps > n_max_steps) {
        const __half2* p = cache + (size_t)(n_max_steps - 1) * FF_GRID;
        cache_bilinear_h2(p, m, c, m, c);
        gap_steps -= n_max_steps;
    }
    if (gap_steps > 0) {
        const __half2* p = cache + (size_t)(gap_steps - 1) * FF_GRID;
        cache_bilinear_h2(p, m, c, m, c);
    }
}

template<bool WRITE_CI>
__global__ __launch_bounds__(256, 4)
void gamma_smc_h2_fwd_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float two_Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const __half2* __restrict__ cache,   // [n_max_steps × FF_GRID]
    int n_max_steps,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        int gap_steps = (int)(pos - prev_pos + 0.5);
        prev_pos = pos;

        if (s > 0 && gap_steps > 0)
            cache_advance_h2(m, c, cache, gap_steps, n_max_steps);

        // Het emission
        int w = s >> 6, bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        long long idx = (long long)s * n_pairs + pid;
        float mean_gen = exp2f(m * 3.321928094887362f) * two_Ne;
        mean_out[idx] = mean_gen;

        if constexpr (WRITE_CI) {
            float alpha = __exp10f(-2.0f * c);
            float inv9a = __frcp_rn(9.0f * alpha);
            float sq = __fsqrt_rn(inv9a);
            float base = 1.0f - inv9a;
            float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
            float hi_f = base + 1.96f * sq;
            lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
            upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
        }
    }
}

// Host launcher — half-precision cached forward-only
void gamma_smc_flow_h2_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache_void, int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;
    float two_Ne = 2.0f * Ne;
    const __half2* d_cache = (const __half2*)d_cache_void;

    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_h2_fwd_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_h2_fwd_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, nullptr, nullptr);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_h2 fwd: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// SYNCHRONIZED forward-only kernel: __syncthreads() per site
// forces all warps in a block to the same loop iteration,
// ensuring they access the same cache layer → L1 hit (~28 cycles)
// instead of random L2 access (~200 cycles).
// ============================================================
template<bool WRITE_CI>
__global__ __launch_bounds__(256)
void gamma_smc_sync_fwd_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float two_Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float2* __restrict__ cache,   // interleaved [n_max_steps × FF_GRID]
    int n_max_steps,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (pid < n_pairs);

    int hi = active ? pair_i[pid] : 0;
    int hj = active ? pair_j[pid] : 0;

    float m = 0.0f, c = 0.0f;
    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        int gap_steps = (int)(pos - prev_pos + 0.5);
        prev_pos = pos;

        // Cache advance with sync to keep warps aligned → L1 coherence
        if (s > 0 && gap_steps > 0) {
            while (gap_steps > n_max_steps) {
                __syncthreads();
                if (active) {
                    const float2* p = cache + (size_t)(n_max_steps - 1) * FF_GRID;
                    cache_bilinear_f2(p, m, c, m, c);
                }
                gap_steps -= n_max_steps;
            }
            __syncthreads();
            if (active) {
                const float2* p = cache + (size_t)(gap_steps - 1) * FF_GRID;
                cache_bilinear_f2(p, m, c, m, c);
            }
        }

        if (active) {
            // Het emission
            int w = s >> 6, bit = s & 63;
            if (w != cur_word) {
                xor_w = packed[(long long)hi * n_words + w]
                      ^ packed[(long long)hj * n_words + w];
                cur_word = w;
            }
            if ((xor_w >> bit) & 1ULL) {
                float alpha = __exp10f(-2.0f * c) + 1.0f;
                float a_log = __log10f(alpha);
                float b_log = -2.0f * c - m;
                m = a_log - b_log;
                c = -0.5f * a_log;
            }

            // Output
            long long idx = (long long)s * n_pairs + pid;
            float mean_gen = exp2f(m * 3.321928094887362f) * two_Ne;
            mean_out[idx] = mean_gen;

            if constexpr (WRITE_CI) {
                float alpha = __exp10f(-2.0f * c);
                float inv9a = __frcp_rn(9.0f * alpha);
                float sq = __fsqrt_rn(inv9a);
                float base = 1.0f - inv9a;
                float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
                float hi_f = base + 1.96f * sq;
                lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
                upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
            }
        }
    }
}

// Host launcher — synchronized forward-only
void gamma_smc_flow_sync_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache_void, int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;
    float two_Ne = 2.0f * Ne;
    const float2* d_cache = (const float2*)d_cache_void;

    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_sync_fwd_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_sync_fwd_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, nullptr, nullptr);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_sync fwd: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// FORWARD-ONLY cached kernel: no backward pass, no forward buffer.
// Outputs mean TMRCA directly. Minimal SFU in hot path.
// ============================================================
template<bool WRITE_CI>
__global__ __launch_bounds__(256, 4)
void gamma_smc_cached_fwd_only_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float two_Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float2* __restrict__ cache,   // interleaved [n_max_steps × FF_GRID]
    int n_max_steps,
    float* __restrict__ mean_out,
    float* __restrict__ lower_out,
    float* __restrict__ upper_out)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    int cur_word = -1;
    uint64_t xor_w = 0;
    double prev_pos = 0.0;

    for (int s = 0; s < S; s++) {
        double pos = positions[s];
        int gap_steps = (int)(pos - prev_pos + 0.5);
        prev_pos = pos;

        if (s > 0 && gap_steps > 0)
            cache_advance_f2(m, c, cache, gap_steps, n_max_steps);

        // Het emission (rare: ~1% of sites)
        int w = s >> 6, bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        // Output: mean_gen = 10^m * 2Ne
        long long idx = (long long)s * n_pairs + pid;
        float mean_gen = exp2f(m * 3.321928094887362f) * two_Ne;
        mean_out[idx] = mean_gen;

        if constexpr (WRITE_CI) {
            float alpha = __exp10f(-2.0f * c);
            float inv9a = __frcp_rn(9.0f * alpha);
            float sq = __fsqrt_rn(inv9a);
            float base = 1.0f - inv9a;
            float lo_f = fmaxf(base - 1.96f * sq, 0.0f);
            float hi_f = base + 1.96f * sq;
            lower_out[idx] = fmaxf(mean_gen * lo_f * lo_f * lo_f, 0.0f);
            upper_out[idx] = mean_gen * hi_f * hi_f * hi_f;
        }
    }
}

// ============================================================
// Host launcher — forward-only cached
// ============================================================
void gamma_smc_flow_cached_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache_void,
    int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out)
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;
    float two_Ne = 2.0f * Ne;
    const float2* d_cache = (const float2*)d_cache_void;

    bool ci = (tmrca_lower_out != nullptr && tmrca_upper_out != nullptr);
    if (ci) {
        gamma_smc_cached_fwd_only_kernel<true><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, tmrca_lower_out, tmrca_upper_out);
    } else {
        gamma_smc_cached_fwd_only_kernel<false><<<grid, block>>>(
            packed, n_words, positions, S, two_Ne,
            pair_i, pair_j, n_pairs,
            d_cache, n_max_steps,
            tmrca_mean_out, nullptr, nullptr);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gamma_smc_cached fwd_only: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// Fused backward + per-site reduction kernel
// Instead of writing [S × n_pairs] mean_out, accumulates per-site
// mean via warp shuffle + atomicAdd. Output is [S] floats.
// Eliminates the massive D2H transfer entirely.
// ============================================================
__global__ void gamma_smc_cached_backward_reduce_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    float Ne,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    const float* __restrict__ cache_mean,
    const float* __restrict__ cache_cv,
    int n_max_steps,
    const float* __restrict__ fwd_mean_in,
    const float* __restrict__ fwd_cv_in,
    float* __restrict__ site_sum,    // [S] — atomicAdd accumulator
    float* __restrict__ site_min,    // [S] or NULL
    float* __restrict__ site_max)    // [S] or NULL
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int hi = pair_i[pid];
    int hj = pair_j[pid];

    float m = 0.0f, c = 0.0f;
    float unscale = 2.0f * Ne;
    int cur_word = -1;
    uint64_t xor_w = 0;

    for (int s = S - 1; s >= 0; s--) {
        float bwd_a = __exp10f(-2.0f * c);
        float bwd_b = __exp10f(-2.0f * c - m);

        long long idx = (long long)s * n_pairs + pid;
        float fm = fwd_mean_in[idx], fc = fwd_cv_in[idx];
        float fwd_a = __exp10f(-2.0f * fc);
        float fwd_b = __exp10f(-2.0f * fc - fm);

        float a_s = fmaxf(fwd_a + bwd_a - 1.0f, 1.0f);
        float b_s = fmaxf(fwd_b + bwd_b - 1.0f, 1e-10f);
        float mean_gen = (a_s / b_s) * unscale;

        // Simple per-thread atomicAdd (no warp shuffle needed)
        atomicAdd(&site_sum[s], mean_gen);


        // Absorb emission for backward propagation
        int w = s >> 6;
        int bit = s & 63;
        if (w != cur_word) {
            xor_w = packed[(long long)hi * n_words + w]
                  ^ packed[(long long)hj * n_words + w];
            cur_word = w;
        }
        if ((xor_w >> bit) & 1ULL) {
            float alpha = __exp10f(-2.0f * c) + 1.0f;
            float a_log = __log10f(alpha);
            float b_log = -2.0f * c - m;
            m = a_log - b_log;
            c = -0.5f * a_log;
        }

        if (s > 0) {
            int gap_steps = (int)(positions[s] - positions[s - 1] + 0.5);
            if (gap_steps > 0)
                cache_advance(m, c, cache_mean, cache_cv, gap_steps, n_max_steps);
        }
    }
}

// Finalize: divide sums by n_pairs to get means
__global__ void finalize_site_mean_kernel(float* site_sum, int S, float inv_n) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S) site_sum[s] *= inv_n;
}

// Host launcher: fused flow_fb + per-site reduction
void gamma_smc_flow_cached_fb_reduce_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const float* d_cache_mean, const float* d_cache_cv,
    int n_max_steps,
    float* fwd_buf,
    float* site_mean_out,  // [S] device
    float* site_min_out,   // [S] device or NULL
    float* site_max_out)   // [S] device or NULL
{
    const int block = 256;
    int grid = (n_pairs + block - 1) / block;

    float* fwd_mean = fwd_buf;
    float* fwd_cv   = fwd_buf + (long long)S * n_pairs;

    // Forward pass (reuse existing kernel)
    gamma_smc_cached_forward_kernel<<<grid, block>>>(
        packed, n_words, positions, S,
        pair_i, pair_j, n_pairs,
        d_cache_mean, d_cache_cv, n_max_steps,
        fwd_mean, fwd_cv);
    cudaDeviceSynchronize();

    // Zero accumulators
    cudaMemset(site_mean_out, 0, S * sizeof(float));

    // Fused backward + reduce (pass NULL for min/max to skip atomic float ops)
    gamma_smc_cached_backward_reduce_kernel<<<grid, block>>>(
        packed, n_words, positions, S, Ne,
        pair_i, pair_j, n_pairs,
        d_cache_mean, d_cache_cv, n_max_steps,
        fwd_mean, fwd_cv,
        site_mean_out, nullptr, nullptr);
    cudaDeviceSynchronize();

    // Finalize: divide by n_pairs
    int fgrid = (S + 255) / 256;
    finalize_site_mean_kernel<<<fgrid, 256>>>(site_mean_out, S, 1.0f / (float)n_pairs);
    cudaDeviceSynchronize();
}
