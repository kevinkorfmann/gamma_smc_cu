#include "gamma_smc_cu/ultrametric.h"
#include <cfloat>
#include <cstdio>
#include <cmath>

// ============================================================
// Helper: pair index for i > j (device)
// ============================================================
__device__ __forceinline__ int d_pair_index(int i, int j) {
    if (i < j) { int t = i; i = j; j = t; }
    return i * (i - 1) / 2 + j;
}

// ============================================================
// Kernel: Ultrametric Projection via Agglomerative Clustering
//
// One block per site. Within each block, threads cooperate to
// run agglomerative clustering on m haplotypes with K time bins.
//
// For m=20, K=32: 190 pairs, 19 merge steps, 6080 candidates
// per step -- fits comfortably in a single block.
//
// Ultrametric constraint enforced: at each merge step, the merge
// time bin must be >= the maximum internal merge time of both
// clusters being merged. Ties broken by preferring lower time bin.
//
// Layout:
//   gamma:   [n_sites][n_pairs][K]  (row-major)
//   output:  [n_sites][n_pairs][K]  messages
//            [n_sites][n_pairs]     assigned bins
// ============================================================

// Maximum subsample size supported (m <= MAX_M)
#define MAX_M 32
#define MAX_PAIRS ((MAX_M * (MAX_M - 1)) / 2)  // 496

__global__ void ultrametric_project_kernel(
    const float* __restrict__ gamma,      // [n_sites x n_pairs x K]
    int m,                                 // number of haplotypes
    int K,                                 // number of time bins
    const double* __restrict__ coal_prior, // [K]
    double damping,                        // alpha
    float* __restrict__ messages_out,      // [n_sites x n_pairs x K]
    int* __restrict__ assigned_bins_out,   // [n_sites x n_pairs]
    int n_sites,
    int n_pairs                            // m*(m-1)/2
) {
    int site = blockIdx.x;
    if (site >= n_sites) return;

    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Shared memory layout:
    //   float posteriors[n_pairs * K]    -- pairwise log-posteriors
    //   int   cluster_id[m]             -- which cluster each haplotype belongs to
    //   int   assigned[n_pairs]          -- assigned time bin per pair
    //   int   active[m]                  -- active cluster flags
    //   int   cluster_max_time[m]        -- max internal merge time per cluster
    //   float coal_prior_sh[K]          -- coalescent prior in shared mem
    //   Reduction workspace for finding best merge
    extern __shared__ char smem[];

    float* posteriors      = (float*)smem;
    int*   cluster_id      = (int*)(posteriors + n_pairs * K);
    int*   assigned        = cluster_id + m;
    int*   active          = assigned + n_pairs;
    int*   cluster_max_t   = active + m;
    float* coal_sh         = (float*)(cluster_max_t + m);
    // Reduction arrays (for block-wide argmax)
    float* best_scores     = coal_sh + K;
    int*   best_ABk        = (int*)(best_scores + n_threads);
    // best_ABk stores encoded (A, B, k) per thread

    // Base offset in global arrays for this site
    int site_base_gamma = site * n_pairs * K;
    int site_base_bins  = site * n_pairs;

    // Load coalescent prior into shared memory
    for (int k = tid; k < K; k += n_threads) {
        coal_sh[k] = (float)coal_prior[k];
    }

    // Load posteriors into shared memory (as log-posteriors)
    for (int idx = tid; idx < n_pairs * K; idx += n_threads) {
        float val = gamma[site_base_gamma + idx];
        posteriors[idx] = (val > 1e-30f) ? logf(val) : -69.0f;  // log(1e-30) ~ -69
    }

    // Initialize cluster assignments and tracking
    if (tid < m) {
        cluster_id[tid] = tid;
        active[tid] = 1;
        cluster_max_t[tid] = -1;  // singletons have no internal merges
    }

    // Initialize assigned bins to -1
    for (int p = tid; p < n_pairs; p += n_threads) {
        assigned[p] = -1;
    }

    __syncthreads();

    // === Agglomerative clustering: m-1 merge steps ===
    for (int merge_step = 0; merge_step < m - 1; merge_step++) {

        // Each thread evaluates a subset of (A, B, k) candidates
        float my_best_score = -FLT_MAX;
        int my_best_A = -1;
        int my_best_B = -1;
        int my_best_k = K;  // start at K so tiebreaking prefers lower k

        int candidate_idx = 0;
        for (int a = 0; a < m; a++) {
            if (!active[a]) continue;
            for (int b = a + 1; b < m; b++) {
                if (!active[b]) continue;

                // Ultrametric constraint: merge time >= max internal time of both clusters
                int k_min = cluster_max_t[a];
                if (cluster_max_t[b] > k_min) k_min = cluster_max_t[b];
                if (k_min < 0) k_min = 0;

                for (int k = k_min; k < K; k++) {
                    if (candidate_idx % n_threads == tid) {
                        // Compute score: sum of log-posteriors for all cross-cluster pairs
                        float score = 0.0f;
                        for (int i = 0; i < m; i++) {
                            if (cluster_id[i] != a) continue;
                            for (int j = 0; j < m; j++) {
                                if (cluster_id[j] != b) continue;
                                int pidx = d_pair_index(i, j);
                                score += posteriors[pidx * K + k];
                            }
                        }
                        // Prefer higher score; break ties by lower k
                        if (score > my_best_score ||
                            (score == my_best_score && k < my_best_k)) {
                            my_best_score = score;
                            my_best_A = a;
                            my_best_B = b;
                            my_best_k = k;
                        }
                    }
                    candidate_idx++;
                }
            }
        }

        // Block-wide reduction to find global best
        best_scores[tid] = my_best_score;
        // Encode (A, B, k) into a single int for reduction
        // A and B < MAX_M (32), k < 64: pack as (A << 20) | (B << 10) | k
        best_ABk[tid] = (my_best_A << 20) | (my_best_B << 10) | (my_best_k & 0x3FF);
        __syncthreads();

        // Parallel reduction: prefer higher score, then lower k for tiebreak
        for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float s1 = best_scores[tid];
                float s2 = best_scores[tid + stride];
                int p1 = best_ABk[tid];
                int p2 = best_ABk[tid + stride];
                int k1 = p1 & 0x3FF;
                int k2 = p2 & 0x3FF;

                if (s2 > s1 || (s2 == s1 && k2 < k1)) {
                    best_scores[tid] = s2;
                    best_ABk[tid] = p2;
                }
            }
            __syncthreads();
        }

        // Thread 0 decodes the winner and broadcasts
        int win_A, win_B, win_k;
        if (tid == 0) {
            int packed = best_ABk[0];
            win_A = (packed >> 20) & 0x3FF;
            win_B = (packed >> 10) & 0x3FF;
            win_k = packed & 0x3FF;
            best_ABk[0] = win_A;
            best_ABk[1] = win_B;
            best_ABk[2] = win_k;
        }
        __syncthreads();

        win_A = best_ABk[0];
        win_B = best_ABk[1];
        win_k = best_ABk[2];

        // Assign time bin to all cross-cluster pairs (A, B) at merge time win_k
        for (int i = tid; i < m; i += n_threads) {
            if (cluster_id[i] == win_A) {
                for (int j = 0; j < m; j++) {
                    if (cluster_id[j] == win_B) {
                        int pidx = d_pair_index(i, j);
                        assigned[pidx] = win_k;
                    }
                }
            }
        }
        __syncthreads();

        // Merge cluster B into A: all haplotypes in B now belong to A
        for (int i = tid; i < m; i += n_threads) {
            if (cluster_id[i] == win_B) {
                cluster_id[i] = win_A;
            }
        }

        // Update cluster max time and deactivate B
        if (tid == 0) {
            cluster_max_t[win_A] = win_k;
            active[win_B] = 0;
        }
        __syncthreads();
    }

    // === Write output: assigned bins and messages ===
    for (int p = tid; p < n_pairs; p += n_threads) {
        int ak = assigned[p];
        assigned_bins_out[site_base_bins + p] = ak;

        for (int k = 0; k < K; k++) {
            float delta_k = (k == ak) ? 1.0f : 0.0f;
            float msg = (float)damping * delta_k + (1.0f - (float)damping) * coal_sh[k];
            messages_out[site_base_gamma + p * K + k] = msg;
        }
    }
}


// ============================================================
// Host launcher
// ============================================================
void ultrametric_project_gpu(
    const float* gamma,
    int n_haplotypes,
    int K,
    const double* coal_prior,
    double damping,
    float* messages_out,
    int* assigned_bins_out,
    int n_sites,
    int n_subsample_pairs
) {
    int m = n_haplotypes;
    int n_pairs = n_subsample_pairs;

    // Block size: 256 threads (sufficient for m<=32, K<=64)
    int block_size = 256;

    // Shared memory calculation
    size_t smem_size = 0;
    smem_size += n_pairs * K * sizeof(float);       // posteriors (log)
    smem_size += m * sizeof(int);                    // cluster_id
    smem_size += n_pairs * sizeof(int);              // assigned
    smem_size += m * sizeof(int);                    // active
    smem_size += m * sizeof(int);                    // cluster_max_t
    smem_size += K * sizeof(float);                  // coal_prior shared
    smem_size += block_size * sizeof(float);         // best_scores
    smem_size += block_size * sizeof(int);           // best_ABk

    // Grid: one block per site
    dim3 grid(n_sites);
    dim3 block(block_size);

    ultrametric_project_kernel<<<grid, block, smem_size>>>(
        gamma, m, K, coal_prior, damping,
        messages_out, assigned_bins_out,
        n_sites, n_pairs
    );
    cudaDeviceSynchronize();
}
