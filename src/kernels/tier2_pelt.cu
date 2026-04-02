#include "tmrca_cu/pelt.h"
#include <cstdio>
#include <cmath>
#include <float.h>

// ============================================================
// Poisson segment cost
// cost(a, b) for the half-open interval [a, b)
//   count = prefix[b] - prefix[a]
//   length = positions[b] - positions[a]
//
// Negative Poisson log-likelihood (up to additive constant):
//   count == 0  =>  2 * mu * length   (expected count under null)
//   count >  0  =>  -count * log(count / length) + count
//
// NOTE: we drop terms that don't depend on the segmentation
// choice; the 2*mu factor only matters when count==0.
// ============================================================

__device__ __forceinline__
double segment_cost(int64_t count, double length, double mu) {
    if (length <= 0.0) return 1e30;
    if (count == 0) {
        return 2.0 * mu * length;
    }
    double c = (double)count;
    double rate = c / length;
    return -c * log(rate) + c;
}

// ============================================================
// PELT kernel — one warp per pair
//
// Each warp runs the full sequential PELT algorithm.
// Lane 0 owns the main DP state; all 32 lanes collaborate
// when evaluating candidate changepoints (the inner loop over
// the pruning set R).
//
// Shared memory layout per warp:
//   double  F[S]          — optimal cost up to site s
//   int     last_cp[S]    — last changepoint index for traceback
//   int     R_set[S]      — current pruning set (candidate starts)
//
// Because S can be very large, we allocate F/last_cp/R_set in
// global memory (one buffer per pair, allocated by the host
// wrapper).
// ============================================================

// We use a two-level approach:
//   1. Coarsen: reduce S to S' = S/stride candidate evaluation points
//   2. Run PELT on the coarsened grid
// This is exact when stride==1 (full PELT).

__global__ void pelt_kernel(
    const int64_t* __restrict__ prefix,   // [n_pairs * S]
    const double*  __restrict__ positions, // [S]
    int S, int n_pairs,
    double mu, double penalty,
    // DP workspace (global memory, [n_pairs * S] each)
    double* F_buf,
    int*    last_cp_buf,
    // Pruning set workspace: we store R as a list per pair.
    // R_buf:      [n_pairs * S]  candidate indices
    // R_size_buf: [n_pairs]      size of R
    int*    R_buf,
    int*    R_size_buf,
    // Outputs
    int*    n_segments_out,
    int*    seg_starts_out,
    int*    seg_ends_out,
    float*  seg_tmrca_out,
    int*    seg_counts_out,
    int     max_segments
) {
    // One warp per pair
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    if (warp_id >= n_pairs) return;

    const int64_t* my_prefix = prefix + (int64_t)warp_id * S;
    double* F       = F_buf       + (int64_t)warp_id * S;
    int*    last_cp = last_cp_buf + (int64_t)warp_id * S;
    int*    R       = R_buf       + (int64_t)warp_id * S;
    int*    R_size  = R_size_buf  + warp_id;

    // ---- Initialization ----
    if (lane == 0) {
        F[0]       = 0.0;
        last_cp[0] = 0;
        R[0]       = 0;
        *R_size    = 1;
    }
    __syncwarp();

    // ---- Main PELT loop ----
    for (int s = 1; s < S; s++) {
        int r_sz = *R_size;

        // Each lane evaluates a subset of candidates in R
        double best_cost = 1e30;
        int    best_cp   = 0;

        for (int idx = lane; idx < r_sz; idx += 32) {
            int t = R[idx];
            int64_t count  = my_prefix[s] - my_prefix[t];
            double  length = positions[s] - positions[t];
            double  c      = segment_cost(count, length, mu);
            double  total  = F[t] + c + penalty;
            if (total < best_cost) {
                best_cost = total;
                best_cp   = t;
            }
        }

        // Warp-level reduction to find global min across lanes
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other_cost = __shfl_down_sync(0xFFFFFFFF, best_cost, offset);
            int    other_cp   = __shfl_down_sync(0xFFFFFFFF, best_cp,   offset);
            if (other_cost < best_cost) {
                best_cost = other_cost;
                best_cp   = other_cp;
            }
        }
        // Broadcast winner from lane 0
        best_cost = __shfl_sync(0xFFFFFFFF, best_cost, 0);
        best_cp   = __shfl_sync(0xFFFFFFFF, best_cp,   0);

        if (lane == 0) {
            F[s]       = best_cost;
            last_cp[s] = best_cp;
        }
        __syncwarp();

        // ---- Pruning step ----
        // Keep candidates t where F[t] + cost(t, s) <= F[s]
        // Also add s as a new candidate.
        // Lane 0 does the pruning (R_size is typically small enough).
        if (lane == 0) {
            int new_sz = 0;
            for (int idx = 0; idx < r_sz; idx++) {
                int t = R[idx];
                int64_t count  = my_prefix[s] - my_prefix[t];
                double  length = positions[s] - positions[t];
                double  c      = segment_cost(count, length, mu);
                if (F[t] + c <= F[s]) {
                    R[new_sz++] = t;
                }
            }
            R[new_sz++] = s;
            *R_size = new_sz;
        }
        __syncwarp();
    }

    // ---- Traceback: recover changepoints ----
    if (lane == 0) {
        // Collect changepoints by backtracking from S-1
        int cp_stack[PELT_MAX_SEGMENTS];
        int n_cp = 0;
        int pos = S - 1;
        while (pos > 0 && n_cp < PELT_MAX_SEGMENTS - 1) {
            cp_stack[n_cp++] = pos;
            pos = last_cp[pos];
        }
        cp_stack[n_cp++] = 0;  // start sentinel

        // cp_stack is in reverse order: [end, ..., 0]
        // Reverse to get segments in order
        int n_seg = n_cp - 1;
        if (n_seg > max_segments) n_seg = max_segments;
        if (n_seg < 1) n_seg = 1;

        int out_base = warp_id * max_segments;
        n_segments_out[warp_id] = n_seg;

        for (int i = 0; i < n_seg; i++) {
            int seg_start = cp_stack[n_cp - 1 - i];
            int seg_end   = cp_stack[n_cp - 2 - i];

            int64_t count  = my_prefix[seg_end] - my_prefix[seg_start];
            double  length = positions[seg_end] - positions[seg_start];

            float tmrca_mle;
            if (count == 0 || length <= 0.0) {
                tmrca_mle = 0.0f;
            } else {
                // MLE: count = 2 * mu * T * length_bp  =>  T = count / (2 * mu * length)
                tmrca_mle = (float)((double)count / (2.0 * mu * length));
            }

            seg_starts_out[out_base + i] = seg_start;
            seg_ends_out[out_base + i]   = seg_end;
            seg_tmrca_out[out_base + i]  = tmrca_mle;
            seg_counts_out[out_base + i] = (int)count;
        }
    }
}


// ============================================================
// Host wrapper
// ============================================================

void pelt_changepoint_gpu(
    const int64_t* prefix,
    const double* positions,
    int S, int n_pairs,
    double mu, double penalty,
    int* n_segments_out,
    int* seg_starts_out,
    int* seg_ends_out,
    float* seg_tmrca_out,
    int* seg_counts_out,
    int max_segments
) {
    // Allocate DP workspace on device
    double* F_buf       = nullptr;
    int*    last_cp_buf = nullptr;
    int*    R_buf       = nullptr;
    int*    R_size_buf  = nullptr;

    size_t dp_elems = (size_t)n_pairs * S;

    cudaMalloc(&F_buf,       dp_elems * sizeof(double));
    cudaMalloc(&last_cp_buf, dp_elems * sizeof(int));
    cudaMalloc(&R_buf,       dp_elems * sizeof(int));
    cudaMalloc(&R_size_buf,  (size_t)n_pairs * sizeof(int));

    // Zero-initialize
    cudaMemset(F_buf,       0, dp_elems * sizeof(double));
    cudaMemset(last_cp_buf, 0, dp_elems * sizeof(int));
    cudaMemset(R_buf,       0, dp_elems * sizeof(int));
    cudaMemset(R_size_buf,  0, (size_t)n_pairs * sizeof(int));

    // Launch: 32 threads per warp, multiple warps per block
    // Use 4 warps (128 threads) per block for occupancy
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * 32;
    int n_blocks = (n_pairs + warps_per_block - 1) / warps_per_block;

    pelt_kernel<<<n_blocks, threads_per_block>>>(
        prefix, positions, S, n_pairs,
        mu, penalty,
        F_buf, last_cp_buf,
        R_buf, R_size_buf,
        n_segments_out, seg_starts_out, seg_ends_out,
        seg_tmrca_out, seg_counts_out, max_segments
    );

    cudaDeviceSynchronize();

    // Free workspace
    cudaFree(F_buf);
    cudaFree(last_cp_buf);
    cudaFree(R_buf);
    cudaFree(R_size_buf);
}
