#include "tmrca_cu/flow_field.h"
#include <cmath>
#include <cstdio>

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
