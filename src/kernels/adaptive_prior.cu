#include "gamma_smc_cu/hmm.h"
#include <cstdio>

// ============================================================
// Kernel: Aggregate posteriors across all pairs and sites
// Computes q_emp[k] = mean_over_{p,s} gamma[p][s][k]
//
// Uses atomic adds to accumulate, then normalizes.
// One thread per (pair_chunk, k) — processes all sites in a loop.
// ============================================================
__global__ void aggregate_posteriors_kernel(
    const float* __restrict__ gamma,  // [n_pairs × S × K]
    int n_pairs, int S, int K,
    double* __restrict__ q_out)       // [K] accumulated sums
{
    int pair_idx = blockIdx.x;
    int k = threadIdx.x;
    if (pair_idx >= n_pairs || k >= K) return;

    double acc = 0.0;
    const float* g = gamma + (long long)pair_idx * S * K;
    for (int s = 0; s < S; s++) {
        acc += (double)g[s * K + k];
    }

    atomicAdd(&q_out[k], acc);
}

void aggregate_posteriors_gpu(const float* gamma, int n_pairs, int S, int K,
                              double* q_empirical_out) {
    // Zero output
    cudaMemset(q_empirical_out, 0, K * sizeof(double));

    // Launch: one block per pair, K threads per block
    aggregate_posteriors_kernel<<<n_pairs, K>>>(gamma, n_pairs, S, K, q_empirical_out);
    cudaDeviceSynchronize();
}

// ============================================================
// Kernel: Blend empirical prior with old prior, then normalize
// q_new[k] = (1-alpha)*q_old[k] + alpha*(q_emp[k] / total_count)
// Then normalize q_new to sum to 1.
// ============================================================
__global__ void blend_prior_kernel(
    const double* __restrict__ q_old,
    const double* __restrict__ q_emp,
    double alpha,
    double inv_count,  // 1.0 / (n_pairs * S)
    int K,
    double* __restrict__ q_new)
{
    int k = threadIdx.x;
    if (k >= K) return;

    double emp_norm = q_emp[k] * inv_count;
    q_new[k] = (1.0 - alpha) * q_old[k] + alpha * emp_norm;
}

// Normalize kernel (single block, K threads)
__global__ void normalize_prior_kernel(double* q, int K) {
    __shared__ double smem[4];  // up to 128/32 = 4 warps

    int k = threadIdx.x;
    if (k >= K) return;

    double val = q[k];

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n_warps = (K + 31) / 32;

    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    double total = 0.0;
    if (warp_id == 0) {
        double v = (lane < n_warps) ? smem[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        }
        total = v;
        if (lane == 0) smem[0] = total;
    }
    __syncthreads();
    total = smem[0];

    if (total > 0.0) {
        q[k] /= total;
    }
}

void blend_prior_gpu(const double* q_old, const double* q_emp, double alpha,
                     int K, double* q_new_out) {
    // We need to know the count — caller passes it through q_emp (un-normalized sums)
    // Actually, the caller will normalize. We accept inv_count = 0 here and
    // the caller handles it. Let's just do the blend + normalize.
    // The caller should have already divided q_emp by count before calling this,
    // OR we can pass inv_count. Let's keep it simple: caller normalizes q_emp on host.

    // Actually, let's just do this properly:
    // blend_prior_gpu assumes q_emp is raw sums, and we'll pass inv_count from host.
    // But that's awkward. Simpler: do blend on host since it's just K doubles.

    // This function is a no-op placeholder — the actual blend is done in the host
    // orchestration function since K is small (32-128 doubles).
}
