#include "gamma_smc_cu/hmm.h"
#include <cmath>
#include <cfloat>
#include <cstdio>

// ============================================================
// Helper: warp-level sum reduction (broadcast to all lanes) — float
// ============================================================
__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// ============================================================
// Helper: warp-level sum reduction (broadcast to all lanes) — double
// ============================================================
__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// ============================================================
// Helper: pair-local sum reduction for K threads — float
// For K=32: pure warp shuffle (no shared memory needed)
// For K=64/128: warp shuffle + shared memory inter-warp reduction
// k_local: thread index within the pair's K threads (0..K-1)
// smem: pointer to this pair's shared memory region (N_WARPS floats)
// ============================================================
template<int K_TPL>
__device__ __forceinline__ float block_reduce_sum_f(float val, float* smem, int k_local) {
    if constexpr (K_TPL == 32) {
        return warp_reduce_sum_f(val);
    } else {
        constexpr int N_WARPS = K_TPL / 32;
        int lane = k_local & 31;
        int warp_id = k_local >> 5;

        val = warp_reduce_sum_f(val);

        if (lane == 0) smem[warp_id] = val;
        __syncthreads();

        float result;
        if (warp_id == 0) {
            float v = (lane < N_WARPS) ? smem[lane] : 0.0f;
            v = warp_reduce_sum_f(v);
            result = v;
            if (lane == 0) smem[0] = result;
        }
        __syncthreads();
        return smem[0];
    }
}

// ============================================================
// Helper: pair-local sum reduction for K threads — double
// ============================================================
template<int K_TPL>
__device__ __forceinline__ double block_reduce_sum(double val, double* smem, int k_local) {
    if constexpr (K_TPL == 32) {
        return warp_reduce_sum(val);
    } else {
        constexpr int N_WARPS = K_TPL / 32;
        int lane = k_local & 31;
        int warp_id = k_local >> 5;

        val = warp_reduce_sum(val);

        if (lane == 0) smem[warp_id] = val;
        __syncthreads();

        double result;
        if (warp_id == 0) {
            double v = (lane < N_WARPS) ? smem[lane] : 0.0;
            v = warp_reduce_sum(v);
            result = v;
            if (lane == 0) smem[0] = result;
        }
        __syncthreads();
        return smem[0];
    }
}

// ============================================================
// Helper: extract XOR bit for pair (i,j) at site s
// ============================================================
__device__ __forceinline__ int get_xor_bit(
    const uint64_t* __restrict__ packed, int n_words,
    int hi, int hj, int site)
{
    int w = site >> 6;        // site / 64
    int bit = site & 63;      // site % 64
    uint64_t wi = packed[(long long)hi * n_words + w];
    uint64_t wj = packed[(long long)hj * n_words + w];
    return (int)((wi ^ wj) >> bit) & 1;
}

// ============================================================
// Kernel: Batched HMM Forward-Backward (templated on K, PPB)
//
// Phase 1: Emission precomputation + fast math intrinsics
// Phase 2: FP32 forward-backward
// Phase 3: Multi-pair blocks (PPB pairs per block)
// Phase 4: Fused summary + EM accumulation (OutputMode)
// Phase 5: Loop unrolling
// ============================================================
template<int K_TPL, int PPB, int MODE>
__global__ void hmm_forward_backward_kernel(
    const uint64_t* __restrict__ packed,
    int n_words,
    const double* __restrict__ positions,
    int S,
    const double* __restrict__ mu,
    const double* __restrict__ cum_rho,
    const double* __restrict__ time_midpoints,
    const double* __restrict__ coal_prior,
    const float* __restrict__ messages,  // [n_pairs × S × K] or NULL
    const int* __restrict__ pair_i_arr,
    const int* __restrict__ pair_j_arr,
    int n_pairs,
    float* __restrict__ gamma_out,        // [n_pairs × S × K] (only for MODE==0)
    double* __restrict__ log_lik_out,
    float* __restrict__ tmrca_mean_out,   // for MODE>=1
    float* __restrict__ tmrca_lower_out,
    float* __restrict__ tmrca_upper_out,
    double* __restrict__ q_accum_out)     // for MODE==2, [K] atomicAdd accumulator
{
    // Shared memory for inter-warp reduction
    // For K>32 with PPB>1, we need per-warp shared mem
    __shared__ float smem_f[K_TPL > 32 ? (K_TPL * PPB / 32) : PPB];

    // Which pair within this block does this thread handle?
    const int lane_in_block = threadIdx.x;
    const int pair_lane = lane_in_block / K_TPL;       // 0..PPB-1
    const int k = lane_in_block % K_TPL;               // time bin 0..K-1

    const int pair_idx = blockIdx.x * PPB + pair_lane;
    if (pair_idx >= n_pairs) return;

    const int hi = pair_i_arr[pair_idx];
    const int hj = pair_j_arr[pair_idx];

    // Pointer to shared memory for this pair's warp reduction
    float* my_smem = smem_f + pair_lane * (K_TPL > 32 ? (K_TPL / 32) : 1);

    // Load time midpoint and prior into registers (double for precision)
    const double t_k_d = time_midpoints[k];
    const float t_k = (float)t_k_d;
    const float q_k = (float)coal_prior[k];

    // ---- Phase 1: Precompute emissions (pair-independent, only depends on k) ----
    // mu is uniform: mu[0] == mu[s] for all s. Load once.
    const float mu0 = (float)mu[0];
    const float emit_match_k   = __expf(-2.0f * mu0 * t_k);     // P(d=0 | bin k)
    const float emit_mismatch_k = 1.0f - emit_match_k;           // P(d=1 | bin k)
    const float gap_base_k     = -2.0f * mu0 * t_k;              // for gap: __expf(gap_base_k * gap_bp)

    // Gamma base pointer for this pair (only used in FULL_GAMMA mode)
    float* gamma = nullptr;
    if constexpr (MODE == 0) {
        gamma = gamma_out + (long long)pair_idx * S * K_TPL;
    }
    const float* msg = messages ? (messages + (long long)pair_idx * S * K_TPL) : nullptr;

    // For SUMMARY_AND_EM mode: accumulate q per bin
    float q_accum = 0.0f;

    // ---- Forward pass alpha storage ----
    // In SUMMARY_ONLY/SUMMARY_AND_EM, we still need alpha for the backward pass.
    // We always write alpha to gamma_out for FULL_GAMMA, or to a dedicated buffer.
    // For fused modes, we still need to store alpha[S][K] somewhere.
    // We reuse gamma_out as alpha storage even in fused modes — we just won't copy it to host.
    float* alpha_store = gamma_out + (long long)pair_idx * S * K_TPL;

    // Cache current XOR word
    int cur_word_idx = -1;
    uint64_t cur_xor_word = 0;

    // ==================== FORWARD PASS ====================

    float alpha_k = q_k;
    if (msg) {
        alpha_k *= msg[0 * K_TPL + k];
    }

    // Emission at site 0
    {
        int w = 0;
        cur_xor_word = packed[(long long)hi * n_words + w] ^ packed[(long long)hj * n_words + w];
        cur_word_idx = w;
        int d = (int)((cur_xor_word >> 0) & 1ULL);
        float emit = d ? emit_mismatch_k : emit_match_k;
        alpha_k *= emit;
    }

    // Normalize
    float sum = block_reduce_sum_f<K_TPL>(alpha_k, my_smem, k);
    if (sum > 0.0f) alpha_k /= sum;
    double log_lik = (sum > 0.0f) ? (double)__logf(sum) : -1e30;

    // Store alpha
    alpha_store[0 * K_TPL + k] = alpha_k;

    // Forward iteration
    #pragma unroll 2
    for (int s = 1; s < S; s++) {
        // Recombination distance
        float r = (float)(cum_rho[s] - cum_rho[s - 1]);

        // Gap emission
        float gap_bp = (float)(positions[s] - positions[s - 1] - 1.0);
        float gap_emit = 1.0f;
        if (gap_bp > 0.0f) {
            gap_emit = __expf(gap_base_k * gap_bp);
        }

        // Transition: decompose into stay + recombine
        float no_recomb = __expf(-r * t_k);
        float stay = no_recomb * alpha_k;
        float recomb_mass = (1.0f - no_recomb) * alpha_k;

        // Block reduction to get total recombination mass
        float total_recomb = block_reduce_sum_f<K_TPL>(recomb_mass, my_smem, k);

        float alpha_new = stay + q_k * total_recomb;

        // EP message
        if (msg) {
            alpha_new *= msg[s * K_TPL + k];
        }

        // Gap emission
        alpha_new *= gap_emit;

        // Site emission
        int site_w = s >> 6;
        int site_bit = s & 63;
        if (site_w != cur_word_idx) {
            cur_xor_word = packed[(long long)hi * n_words + site_w]
                         ^ packed[(long long)hj * n_words + site_w];
            cur_word_idx = site_w;
        }
        int d = (int)((cur_xor_word >> site_bit) & 1ULL);
        float emit = d ? emit_mismatch_k : emit_match_k;
        alpha_new *= emit;

        // Normalize
        sum = block_reduce_sum_f<K_TPL>(alpha_new, my_smem, k);
        if (sum > 0.0f) alpha_new /= sum;
        log_lik += (sum > 0.0f) ? (double)__logf(sum) : -1e30;

        alpha_k = alpha_new;
        alpha_store[s * K_TPL + k] = alpha_k;
    }

    // ==================== BACKWARD PASS ====================

    float beta_k = 1.0f;

    // Handle last site gamma/summary (backward starts from S-2, so site S-1 gamma = alpha * 1.0 = alpha)
    if constexpr (MODE >= 1) {
        // Site S-1: gamma = alpha (beta=1, normalized)
        float gamma_val = alpha_store[(S - 1) * K_TPL + k];
        float mean_contrib = gamma_val * t_k;
        float site_mean = block_reduce_sum_f<K_TPL>(mean_contrib, my_smem, k);

        // CI: thread 0 scans CDF
        float lower_val, upper_val;
        if constexpr (K_TPL == 32) {
            // Warp-level CDF scan using shuffles
            float cum = 0.0f;
            lower_val = (float)time_midpoints[0];
            upper_val = (float)time_midpoints[K_TPL - 1];
            bool lower_set = false;
            for (int kk = 0; kk < K_TPL; kk++) {
                float gk = __shfl_sync(0xFFFFFFFF, gamma_val, kk);
                cum += gk;
                if (!lower_set && cum >= 0.025f) {
                    lower_val = (float)time_midpoints[kk];
                    lower_set = true;
                }
                if (cum >= 0.975f) {
                    upper_val = (float)time_midpoints[kk];
                    break;
                }
            }
        } else {
            // For K>32, use shared memory approach
            lower_val = (float)time_midpoints[0];
            upper_val = (float)time_midpoints[K_TPL - 1];
            // Fallback: thread 0 reads from alpha_store
            // (gamma_val at S-1 = alpha)
        }

        long long out_idx = (long long)pair_idx * S + (S - 1);
        if (k == 0) {
            tmrca_mean_out[out_idx] = site_mean;
            tmrca_lower_out[out_idx] = lower_val;
            tmrca_upper_out[out_idx] = upper_val;
        }
        if constexpr (MODE == 2) {
            q_accum += gamma_val;
        }
    }

    #pragma unroll 2
    for (int s = S - 2; s >= 0; s--) {
        float r = (float)(cum_rho[s + 1] - cum_rho[s]);

        // Emission at site s+1
        int site_w = (s + 1) >> 6;
        int site_bit = (s + 1) & 63;
        if (site_w != cur_word_idx) {
            cur_xor_word = packed[(long long)hi * n_words + site_w]
                         ^ packed[(long long)hj * n_words + site_w];
            cur_word_idx = site_w;
        }
        int d_next = (int)((cur_xor_word >> site_bit) & 1ULL);
        float emit_next = d_next ? emit_mismatch_k : emit_match_k;

        // Gap emission
        float gap_bp = (float)(positions[s + 1] - positions[s] - 1.0);
        float gap_emit = 1.0f;
        if (gap_bp > 0.0f) {
            gap_emit = __expf(gap_base_k * gap_bp);
        }

        float be = beta_k * emit_next * gap_emit;

        float q_be = q_k * be;
        float total_q_be = block_reduce_sum_f<K_TPL>(q_be, my_smem, k);

        float no_recomb = __expf(-r * t_k);
        float beta_new = no_recomb * be + (1.0f - no_recomb) * total_q_be;

        // Normalize
        sum = block_reduce_sum_f<K_TPL>(beta_new, my_smem, k);
        if (sum > 0.0f) beta_new /= sum;

        beta_k = beta_new;

        // Compute gamma = alpha * beta, normalized
        float alpha_s = alpha_store[s * K_TPL + k];
        float gamma_val = alpha_s * beta_k;
        float gamma_sum = block_reduce_sum_f<K_TPL>(gamma_val, my_smem, k);
        if (gamma_sum > 0.0f) gamma_val /= gamma_sum;

        if constexpr (MODE == 0) {
            gamma[s * K_TPL + k] = gamma_val;
        }

        if constexpr (MODE >= 1) {
            // Fused summary: compute mean TMRCA
            float mean_contrib = gamma_val * t_k;
            float site_mean = block_reduce_sum_f<K_TPL>(mean_contrib, my_smem, k);

            // CI: CDF scan
            float lower_val, upper_val;
            if constexpr (K_TPL == 32) {
                float cum = 0.0f;
                lower_val = (float)time_midpoints[0];
                upper_val = (float)time_midpoints[K_TPL - 1];
                bool lower_set = false;
                for (int kk = 0; kk < K_TPL; kk++) {
                    float gk = __shfl_sync(0xFFFFFFFF, gamma_val, kk);
                    cum += gk;
                    if (!lower_set && cum >= 0.025f) {
                        lower_val = (float)time_midpoints[kk];
                        lower_set = true;
                    }
                    if (cum >= 0.975f) {
                        upper_val = (float)time_midpoints[kk];
                        break;
                    }
                }
            } else {
                lower_val = (float)time_midpoints[0];
                upper_val = (float)time_midpoints[K_TPL - 1];
                // For K>32, write gamma temporarily and let thread 0 scan
                // We use alpha_store as scratch for this
                alpha_store[s * K_TPL + k] = gamma_val;
                __syncthreads();
                if (k == 0) {
                    float cum = 0.0f;
                    bool lower_set_local = false;
                    for (int kk = 0; kk < K_TPL; kk++) {
                        cum += alpha_store[s * K_TPL + kk];
                        if (!lower_set_local && cum >= 0.025f) {
                            lower_val = (float)time_midpoints[kk];
                            lower_set_local = true;
                        }
                        if (cum >= 0.975f) {
                            upper_val = (float)time_midpoints[kk];
                            break;
                        }
                    }
                }
            }

            long long out_idx = (long long)pair_idx * S + s;
            if (k == 0) {
                tmrca_mean_out[out_idx] = site_mean;
                tmrca_lower_out[out_idx] = lower_val;
                tmrca_upper_out[out_idx] = upper_val;
            }

            if constexpr (MODE == 2) {
                q_accum += gamma_val;
            }
        }
    }

    // Store log-likelihood
    if (k == 0) {
        log_lik_out[pair_idx] = log_lik;
    }

    // EM accumulation
    if constexpr (MODE == 2) {
        atomicAdd(&q_accum_out[k], (double)q_accum);
    }
}

// ============================================================
// Host launcher helpers
// ============================================================

// Compute grid/block dims for given K, PPB
template<int K_TPL, int PPB>
static inline void get_launch_config(int n_pairs, int& grid, int& block) {
    block = K_TPL * PPB;
    grid = (n_pairs + PPB - 1) / PPB;
}

// Dispatch on mode
template<int K_TPL, int PPB>
static void launch_kernel(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    const double* mu, const double* cum_rho,
    const double* time_midpoints, const double* coal_prior,
    const float* messages,
    const int* pair_i, const int* pair_j, int n_pairs,
    float* gamma_out, double* log_lik_out,
    float* tmrca_mean_out, float* tmrca_lower_out, float* tmrca_upper_out,
    double* q_accum_out,
    HMMOutputMode mode)
{
    int grid, block;
    get_launch_config<K_TPL, PPB>(n_pairs, grid, block);

    switch (mode) {
        case FULL_GAMMA:
            hmm_forward_backward_kernel<K_TPL, PPB, 0><<<grid, block>>>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out);
            break;
        case SUMMARY_ONLY:
            hmm_forward_backward_kernel<K_TPL, PPB, 1><<<grid, block>>>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out);
            break;
        case SUMMARY_AND_EM:
            hmm_forward_backward_kernel<K_TPL, PPB, 2><<<grid, block>>>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out);
            break;
    }
    cudaDeviceSynchronize();
}

// ============================================================
// Public API: hmm_forward_backward_gpu (new signature)
// ============================================================
void hmm_forward_backward_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    const double* mu, const double* cum_rho,
    const double* time_midpoints, const double* coal_prior,
    const float* messages,
    const int* pair_i, const int* pair_j, int n_pairs,
    float* gamma_out, double* log_lik_out,
    float* tmrca_mean_out, float* tmrca_lower_out, float* tmrca_upper_out,
    double* q_accum_out,
    int K, HMMOutputMode mode)
{
    // Phase 3: PPB=4 for K=32, PPB=2 for K=64, PPB=1 for K=128
    switch (K) {
        case 32:
            launch_kernel<32, 4>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out, mode);
            break;
        case 64:
            launch_kernel<64, 2>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out, mode);
            break;
        case 128:
            launch_kernel<128, 1>(
                packed, n_words, positions, S,
                mu, cum_rho, time_midpoints, coal_prior,
                messages, pair_i, pair_j, n_pairs,
                gamma_out, log_lik_out,
                tmrca_mean_out, tmrca_lower_out, tmrca_upper_out,
                q_accum_out, mode);
            break;
        default:
            break;
    }
}

// ============================================================
// Kernel: Extract posterior summaries (runtime K) — kept for
// backward compat when FULL_GAMMA mode is used
// ============================================================
__global__ void extract_summaries_kernel(
    const float* __restrict__ gamma,
    int n_pairs, int S, int K,
    const double* __restrict__ time_midpoints,
    float* __restrict__ tmrca_mean,
    float* __restrict__ tmrca_lower,
    float* __restrict__ tmrca_upper)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx / S;
    int s = idx % S;

    if (pair_idx >= n_pairs) return;

    const float* g = gamma + ((long long)pair_idx * S + s) * K;

    double mean = 0.0;
    double cum = 0.0;
    double lower = time_midpoints[0];
    double upper = time_midpoints[K - 1];
    bool lower_set = false;

    for (int k = 0; k < K; k++) {
        double gk = (double)g[k];
        double tk = time_midpoints[k];
        mean += gk * tk;
        cum += gk;
        if (!lower_set && cum >= 0.025) {
            lower = tk;
            lower_set = true;
        }
        if (cum >= 0.975) {
            upper = tk;
            break;
        }
    }

    long long out_idx = (long long)pair_idx * S + s;
    tmrca_mean[out_idx] = (float)mean;
    tmrca_lower[out_idx] = (float)lower;
    tmrca_upper[out_idx] = (float)upper;
}

void extract_summaries_gpu(const float* gamma, int n_pairs, int S,
                           const double* time_midpoints,
                           float* tmrca_mean, float* tmrca_lower,
                           float* tmrca_upper,
                           int K) {
    int total = n_pairs * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    extract_summaries_kernel<<<grid, block>>>(
        gamma, n_pairs, S, K, time_midpoints,
        tmrca_mean, tmrca_lower, tmrca_upper);
    cudaDeviceSynchronize();
}
