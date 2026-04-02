#include "tmrca_cu/hmm.h"
#include <cstdio>

// ============================================================
// Kernel: Compute Site Frequency Spectrum
// Each thread handles one site, counts alleles across haplotypes.
// For large n: transpose-based approach with warp-level popcount.
// ============================================================

// Simple version: each thread iterates over haplotypes for one site
__global__ void sfs_kernel_simple(const uint64_t* __restrict__ packed,
                                  int n, int S, int n_words,
                                  int* __restrict__ sfs) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= S) return;

    int w = site / 64;
    int bit = site % 64;

    int count = 0;
    for (int hap = 0; hap < n; hap++) {
        count += (int)((packed[(long long)hap * n_words + w] >> bit) & 1ULL);
    }
    atomicAdd(&sfs[count], 1);
}

// Optimized: process 64 sites per warp using transposed access
// For each word position, all 64 sites share the same memory access pattern
__global__ void sfs_kernel_fast(const uint64_t* __restrict__ packed,
                                int n, int S, int n_words,
                                int* __restrict__ sfs) {
    int word_idx = blockIdx.x;  // which 64-site group
    if (word_idx >= n_words) return;

    int lane = threadIdx.x;  // 0..31 within warp
    int warp_id = threadIdx.y;  // which bit-pair this warp handles
    // Each warp handles 2 bits (so 32 warps for 64 bits, or we loop)

    // Simple approach: each thread in a warp processes a chunk of haplotypes
    // and we reduce counts
    for (int bit = warp_id; bit < 64; bit += blockDim.y) {
        int site = word_idx * 64 + bit;
        if (site >= S) continue;

        int count = 0;
        // Each thread processes n/32 haplotypes
        for (int hap = lane; hap < n; hap += 32) {
            count += (int)((packed[(long long)hap * n_words + word_idx] >> bit) & 1ULL);
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            count += __shfl_down_sync(0xFFFFFFFF, count, offset);
        }

        if (lane == 0) {
            atomicAdd(&sfs[count], 1);
        }
    }
}

void compute_sfs_gpu(const uint64_t* packed, int n, int S, int n_words,
                     int* sfs_out) {
    // Zero the SFS array
    cudaMemset(sfs_out, 0, (n + 1) * sizeof(int));

    if (n <= 256) {
        // Use simple kernel for small n
        int block = 256;
        int grid = (S + block - 1) / block;
        sfs_kernel_simple<<<grid, block>>>(packed, n, S, n_words, sfs_out);
    } else {
        // Use fast kernel with warp-level reduction
        // One block per word, blockDim = (32, min(64, some limit))
        dim3 block(32, 8);  // 8 warps per block, each handles 8 bits per iteration
        int grid = n_words;
        sfs_kernel_fast<<<grid, block>>>(packed, n, S, n_words, sfs_out);
    }
    cudaDeviceSynchronize();
}
