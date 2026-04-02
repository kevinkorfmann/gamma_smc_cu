#include "tmrca_cu/bitpack.h"
#include <cstdio>

// ============================================================
// Kernel: Per-site nucleotide diversity pi(s)
//
// For a sample of pairs, compute:
//   pi(s) = (1 / n_pairs) * sum_{(i,j)} d_ij(s)
// where d_ij(s) = XOR bit at site s.
//
// We accumulate across pairs using atomicAdd on a per-site counter.
// ============================================================
__global__ void site_pi_kernel(
    const uint64_t* __restrict__ packed,
    int n_words, int S,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    int n_pairs,
    float* __restrict__ pi_out)  // [S], atomically accumulated
{
    // Each block handles one pair, threads handle sites in stripes
    int pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    int hi = pair_i[pair_idx];
    int hj = pair_j[pair_idx];

    const uint64_t* row_i = packed + (long long)hi * n_words;
    const uint64_t* row_j = packed + (long long)hj * n_words;

    // Each thread processes multiple words
    for (int w = threadIdx.x; w < n_words; w += blockDim.x) {
        uint64_t xor_word = row_i[w] ^ row_j[w];
        int base_site = w * 64;

        // Unpack each bit and atomically add to pi
        while (xor_word != 0) {
            int bit = __ffsll(xor_word) - 1;  // find lowest set bit
            int site = base_site + bit;
            if (site < S) {
                atomicAdd(&pi_out[site], 1.0f);
            }
            xor_word &= xor_word - 1;  // clear lowest set bit
        }
    }
}

void site_pi_gpu(const uint64_t* packed, int n_words, int S,
                 const int* pair_i, const int* pair_j, int n_pairs,
                 float* pi_out) {
    // Zero output
    cudaMemset(pi_out, 0, S * sizeof(float));

    int block = 128;
    site_pi_kernel<<<n_pairs, block>>>(packed, n_words, S,
                                        pair_i, pair_j, n_pairs, pi_out);
    cudaDeviceSynchronize();
}

// ============================================================
// Kernel: Multi-scale windowed divergence
// For each pair and site, compute divergence at multiple window scales
// div_out[pair][scale][site] = (prefix[s+W] - prefix[s-W]) / (2*mu*(positions[s+W]-positions[s-W]))
// ============================================================
__global__ void multiscale_divergence_kernel(
    const int64_t* __restrict__ prefix,
    const double* __restrict__ positions,
    int S, int n_pairs,
    const int* __restrict__ window_sizes,  // [n_scales] in number of sites
    int n_scales,
    double mu,
    float* __restrict__ div_out)  // [n_pairs × n_scales × S]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pairs * S;
    if (idx >= total) return;

    int pair_idx = idx / S;
    int s = idx % S;

    const int64_t* p = prefix + (long long)pair_idx * S;

    for (int sc = 0; sc < n_scales; sc++) {
        int W = window_sizes[sc];
        int left = s - W;
        int right = s + W;
        if (left < 0) left = 0;
        if (right >= S) right = S - 1;

        int64_t diff_count = p[right] - (left > 0 ? p[left] : 0);
        double span_bp = positions[right] - positions[left];
        double tmrca_est = 0.0;
        if (span_bp > 0.0 && mu > 0.0) {
            tmrca_est = (double)diff_count / (2.0 * mu * span_bp);
        }

        long long out_idx = ((long long)pair_idx * n_scales + sc) * S + s;
        div_out[out_idx] = (float)tmrca_est;
    }
}

void multiscale_divergence_gpu(
    const int64_t* prefix,
    const double* positions,
    int S, int n_pairs,
    const int* d_window_sizes, int n_scales,
    double mu,
    float* div_out)
{
    int total = n_pairs * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    multiscale_divergence_kernel<<<grid, block>>>(
        prefix, positions, S, n_pairs,
        d_window_sizes, n_scales, mu, div_out);
    cudaDeviceSynchronize();
}
