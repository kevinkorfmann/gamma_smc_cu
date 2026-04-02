#include "tmrca_cu/bitpack.h"
#include <cstdio>

// ============================================================
// Kernel: Bitpack genotype matrix
// Each thread packs 64 consecutive sites into one uint64 word
// for one haplotype.
// Grid: (ceil(n_words/4), ceil(n/256))
// Block: (4, 256) -> 4 words × 256 haplotypes
// ============================================================
__global__ void bitpack_kernel(const uint8_t* __restrict__ G,
                               uint64_t* __restrict__ packed,
                               int n, int S, int n_words) {
    int hap = blockIdx.y * blockDim.y + threadIdx.y;
    int w   = blockIdx.x * blockDim.x + threadIdx.x;

    if (hap >= n || w >= n_words) return;

    uint64_t word = 0;
    int base_site = w * 64;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
        int site = base_site + bit;
        if (site < S) {
            uint64_t val = (uint64_t)G[hap * S + site];
            word |= (val & 1ULL) << bit;
        }
    }
    packed[hap * n_words + w] = word;
}

void bitpack_genotypes_gpu(const uint8_t* G, uint64_t* packed,
                           int n, int S, int n_words) {
    dim3 block(4, 256);
    dim3 grid((n_words + 3) / 4, (n + 255) / 256);
    bitpack_kernel<<<grid, block>>>(G, packed, n, S, n_words);
    cudaDeviceSynchronize();
}

// ============================================================
// Kernel: Unpack (for validation)
// ============================================================
__global__ void unpack_kernel(const uint64_t* __restrict__ packed,
                              uint8_t* __restrict__ G,
                              int n, int S, int n_words) {
    int hap = blockIdx.y * blockDim.y + threadIdx.y;
    int w   = blockIdx.x * blockDim.x + threadIdx.x;

    if (hap >= n || w >= n_words) return;

    uint64_t word = packed[hap * n_words + w];
    int base_site = w * 64;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
        int site = base_site + bit;
        if (site < S) {
            G[hap * S + site] = (uint8_t)((word >> bit) & 1ULL);
        }
    }
}

void unpack_genotypes_gpu(const uint64_t* packed, uint8_t* G,
                          int n, int S, int n_words) {
    dim3 block(4, 256);
    dim3 grid((n_words + 3) / 4, (n + 255) / 256);
    unpack_kernel<<<grid, block>>>(packed, G, n, S, n_words);
    cudaDeviceSynchronize();
}

// ============================================================
// Kernel: Pairwise XOR prefix scan
// One warp (32 threads) per pair. Each thread processes a
// stripe of words, computing cumulative popcount.
// ============================================================
__global__ void prefix_scan_kernel(const uint64_t* __restrict__ packed,
                                   int n_words, int S,
                                   const int* __restrict__ pair_i,
                                   const int* __restrict__ pair_j,
                                   int n_pairs,
                                   int64_t* __restrict__ prefix_out) {
    int pair_idx = blockIdx.x;
    int lane = threadIdx.x;  // 0..31

    if (pair_idx >= n_pairs) return;

    int hi = pair_i[pair_idx];
    int hj = pair_j[pair_idx];

    const uint64_t* row_i = packed + (long long)hi * n_words;
    const uint64_t* row_j = packed + (long long)hj * n_words;
    int64_t* out = prefix_out + (long long)pair_idx * S;

    // Process words in warp-striped order, compute per-word popcount
    // Then do a cumulative sum across the warp using shuffle

    int64_t running = 0;

    for (int w_base = 0; w_base < n_words; w_base += 32) {
        int w = w_base + lane;
        int count = 0;
        if (w < n_words) {
            uint64_t xor_word = row_i[w] ^ row_j[w];
            count = __popcll(xor_word);
        }

        // Inclusive prefix scan within warp using shuffle
        int val = count;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n_val = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane >= offset) val += n_val;
        }
        // val now holds inclusive prefix sum within this chunk of 32 words
        // Add running total
        val += (int)running;

        // Write per-site cumulative counts
        if (w < n_words) {
            uint64_t xor_word = row_i[w] ^ row_j[w];
            int base_site = w * 64;
            int local_cum = val - count;  // exclusive prefix for this word
            for (int bit = 0; bit < 64 && base_site + bit < S; bit++) {
                local_cum += (int)((xor_word >> bit) & 1ULL);
                out[base_site + bit] = (int64_t)local_cum;
            }
        }

        // Update running total: broadcast from last lane
        int total_chunk = __shfl_sync(0xFFFFFFFF, val, 31);
        running = (int64_t)total_chunk;
    }
}

void pairwise_prefix_scan_gpu(const uint64_t* packed, int n_words, int S,
                              const int* pair_i, const int* pair_j,
                              int n_pairs, int64_t* prefix_out) {
    // One block per pair, 32 threads (one warp)
    prefix_scan_kernel<<<n_pairs, 32>>>(packed, n_words, S,
                                        pair_i, pair_j, n_pairs,
                                        prefix_out);
    cudaDeviceSynchronize();
}

// ============================================================
// Kernel: Windowed divergence
// div[pair][s] = prefix[s + W] - prefix[max(0, s - W)]
// for a window of W sites on each side
// ============================================================
__global__ void windowed_div_kernel(const int64_t* __restrict__ prefix,
                                    int S, int n_pairs, int W,
                                    float* __restrict__ div_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx / S;
    int s = idx % S;

    if (pair_idx >= n_pairs) return;

    const int64_t* p = prefix + (long long)pair_idx * S;

    int left = s - W;
    int right = s + W;
    if (right >= S) right = S - 1;

    int64_t right_val = p[right];
    int64_t left_val = (left >= 0) ? p[left] : 0;

    div_out[(long long)pair_idx * S + s] = (float)(right_val - left_val);
}

void windowed_divergence_gpu(const int64_t* prefix, int S,
                             int n_pairs, int window_sites,
                             float* div_out) {
    int total = n_pairs * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    windowed_div_kernel<<<grid, block>>>(prefix, S, n_pairs, window_sites,
                                         div_out);
    cudaDeviceSynchronize();
}
