#include "gamma_smc_cu/ultrametric.h"
#include <cfloat>
#include <cstdio>
#include <cmath>

// ============================================================
// Kernel: Convergence Check via Max-Reduction
//
// Computes max |messages[i] - messages_prev[i]| across all
// elements. Uses warp shuffle + shared memory two-level
// reduction for efficiency.
//
// Grid: standard reduction grid (one block per chunk)
// Block: 256 threads
// ============================================================

#define CONV_BLOCK_SIZE 256

__global__ void convergence_reduce_kernel(
    const float* __restrict__ messages,
    const float* __restrict__ messages_prev,
    int total_elements,
    float* __restrict__ partial_max   // [gridDim.x]
) {
    __shared__ float sdata[CONV_BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float my_max = 0.0f;

    // Each thread processes two elements initially (grid-stride)
    if (gid < total_elements) {
        my_max = fabsf(messages[gid] - messages_prev[gid]);
    }
    if (gid + blockDim.x < total_elements) {
        float val = fabsf(messages[gid + blockDim.x] - messages_prev[gid + blockDim.x]);
        if (val > my_max) my_max = val;
    }

    // Grid-stride loop for large arrays
    int stride = gridDim.x * blockDim.x * 2;
    for (int idx = gid + stride; idx < total_elements; idx += stride) {
        float val = fabsf(messages[idx] - messages_prev[idx]);
        if (val > my_max) my_max = val;
        if (idx + blockDim.x < total_elements) {
            val = fabsf(messages[idx + blockDim.x] - messages_prev[idx + blockDim.x]);
            if (val > my_max) my_max = val;
        }
    }

    sdata[tid] = my_max;
    __syncthreads();

    // Shared memory reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* vs = sdata;
        if (vs[tid + 32] > vs[tid]) vs[tid] = vs[tid + 32];
        if (vs[tid + 16] > vs[tid]) vs[tid] = vs[tid + 16];
        if (vs[tid +  8] > vs[tid]) vs[tid] = vs[tid +  8];
        if (vs[tid +  4] > vs[tid]) vs[tid] = vs[tid +  4];
        if (vs[tid +  2] > vs[tid]) vs[tid] = vs[tid +  2];
        if (vs[tid +  1] > vs[tid]) vs[tid] = vs[tid +  1];
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

// Second-stage reduction: reduce partial maxima to a single scalar
__global__ void convergence_final_kernel(
    const float* __restrict__ partial_max,
    int n_blocks,
    float* __restrict__ max_delta_out
) {
    __shared__ float sdata[CONV_BLOCK_SIZE];

    int tid = threadIdx.x;
    float my_max = 0.0f;

    for (int i = tid; i < n_blocks; i += blockDim.x) {
        float val = partial_max[i];
        if (val > my_max) my_max = val;
    }

    sdata[tid] = my_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vs = sdata;
        if (vs[tid + 32] > vs[tid]) vs[tid] = vs[tid + 32];
        if (vs[tid + 16] > vs[tid]) vs[tid] = vs[tid + 16];
        if (vs[tid +  8] > vs[tid]) vs[tid] = vs[tid +  8];
        if (vs[tid +  4] > vs[tid]) vs[tid] = vs[tid +  4];
        if (vs[tid +  2] > vs[tid]) vs[tid] = vs[tid +  2];
        if (vs[tid +  1] > vs[tid]) vs[tid] = vs[tid +  1];
    }

    if (tid == 0) {
        max_delta_out[0] = sdata[0];
    }
}


// ============================================================
// Host launcher
// ============================================================
void check_convergence_gpu(
    const float* messages,
    const float* messages_prev,
    int total_elements,
    float* max_delta_out
) {
    int block_size = CONV_BLOCK_SIZE;
    int n_blocks = (total_elements + block_size * 2 - 1) / (block_size * 2);
    if (n_blocks < 1) n_blocks = 1;
    // Cap blocks to avoid excessive partial_max allocation
    if (n_blocks > 4096) n_blocks = 4096;

    // Allocate temporary partial max array
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, n_blocks * sizeof(float));

    // First pass: per-block max reduction
    convergence_reduce_kernel<<<n_blocks, block_size>>>(
        messages, messages_prev, total_elements, d_partial
    );

    // Second pass: reduce partial maxima to single value
    convergence_final_kernel<<<1, block_size>>>(
        d_partial, n_blocks, max_delta_out
    );

    cudaDeviceSynchronize();
    cudaFree(d_partial);
}
