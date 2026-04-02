#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "tmrca_cu/api.h"

namespace py = pybind11;

// Helper: check CUDA error
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while(0)

// ============================================================
// Bitpack / Unpack
// ============================================================
py::array_t<uint64_t> py_bitpack(py::array_t<uint8_t, py::array::c_style> G) {
    auto buf = G.request();
    if (buf.ndim != 2)
        throw std::runtime_error("G must be 2D (n x S)");

    int n = (int)buf.shape[0];
    int S = (int)buf.shape[1];
    int n_words = (S + 63) / 64;

    uint8_t* d_G;
    uint64_t* d_packed;
    size_t G_size = (size_t)n * S;
    size_t packed_size = (size_t)n * n_words;

    CUDA_CHECK(cudaMalloc(&d_G, G_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, packed_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, buf.ptr, G_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);

    auto result = py::array_t<uint64_t>({n, n_words});
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_packed,
                          packed_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaFree(d_G);
    cudaFree(d_packed);
    return result;
}

py::array_t<uint8_t> py_unpack(py::array_t<uint64_t, py::array::c_style> packed,
                                int n, int S) {
    auto buf = packed.request();
    int n_words = (int)buf.shape[1];

    uint64_t* d_packed;
    uint8_t* d_G;
    size_t packed_size = (size_t)n * n_words;
    size_t G_size = (size_t)n * S;

    CUDA_CHECK(cudaMalloc(&d_packed, packed_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_G, G_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_packed, buf.ptr, packed_size * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    unpack_genotypes_gpu(d_packed, d_G, n, S, n_words);

    auto result = py::array_t<uint8_t>({n, S});
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_G,
                          G_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_G);
    return result;
}

// ============================================================
// Pairwise prefix scan
// ============================================================
py::array_t<int64_t> py_pairwise_prefix_scan(
    py::array_t<uint8_t, py::array::c_style> G,
    std::vector<std::pair<int, int>> pairs)
{
    auto buf = G.request();
    int n = (int)buf.shape[0];
    int S = (int)buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    // Bitpack on GPU
    uint8_t* d_G;
    uint64_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    int* d_pi; int* d_pj;
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    int64_t* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)n_pairs * S * sizeof(int64_t)));

    pairwise_prefix_scan_gpu(d_packed, n_words, S, d_pi, d_pj, n_pairs, d_prefix);

    auto result = py::array_t<int64_t>({n_pairs, S});
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_prefix,
                          (size_t)n_pairs * S * sizeof(int64_t), cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_pi);
    cudaFree(d_pj);
    cudaFree(d_prefix);
    return result;
}

// ============================================================
// Windowed divergence
// ============================================================
py::array_t<float> py_windowed_divergence(
    py::array_t<uint8_t, py::array::c_style> G,
    std::vector<std::pair<int, int>> pairs,
    int window_sites)
{
    auto buf = G.request();
    int n = (int)buf.shape[0];
    int S = (int)buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    // Bitpack
    uint8_t* d_G;
    uint64_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    int* d_pi; int* d_pj;
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Prefix scan
    int64_t* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)n_pairs * S * sizeof(int64_t)));
    pairwise_prefix_scan_gpu(d_packed, n_words, S, d_pi, d_pj, n_pairs, d_prefix);

    // Windowed divergence
    float* d_div;
    CUDA_CHECK(cudaMalloc(&d_div, (size_t)n_pairs * S * sizeof(float)));
    windowed_divergence_gpu(d_prefix, S, n_pairs, window_sites, d_div);

    auto result = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_div,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_pi);
    cudaFree(d_pj);
    cudaFree(d_prefix);
    cudaFree(d_div);
    return result;
}

// ============================================================
// SFS
// ============================================================
py::array_t<int> py_compute_sfs(py::array_t<uint8_t, py::array::c_style> G) {
    auto buf = G.request();
    int n = (int)buf.shape[0];
    int S = (int)buf.shape[1];
    int n_words = (S + 63) / 64;

    // Bitpack
    uint8_t* d_G;
    uint64_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    // SFS
    int* d_sfs;
    CUDA_CHECK(cudaMalloc(&d_sfs, (n + 1) * sizeof(int)));
    compute_sfs_gpu(d_packed, n, S, n_words, d_sfs);

    auto result = py::array_t<int>(n + 1);
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_sfs,
                          (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_sfs);
    return result;
}

// ============================================================
// Coalescent prior
// ============================================================
py::array_t<double> py_coalescent_prior(double Ne, int K_bins, double t_max) {
    if (t_max <= 0) t_max = 10.0 * Ne;
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);
    auto result = py::array_t<double>(K_bins);
    std::memcpy(result.mutable_data(), prior.data(), K_bins * sizeof(double));
    return result;
}

// ============================================================
// HMM posterior
// ============================================================
py::array_t<float> py_hmm_posterior(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::pair<int, int> pair,
    int K_bins,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    double t_max)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;

    if (K_bins != 32 && K_bins != 64 && K_bins != 128)
        throw std::runtime_error("K must be 32, 64, or 128");

    if (t_max <= 0) t_max = 10.0 * Ne;

    // Coalescent prior
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);

    // Mu and rho arrays (uniform)
    std::vector<double> mu_arr(S, mu_scalar);
    std::vector<double> cum_rho_arr(S);
    double* pos_ptr = (double*)pos_buf.ptr;
    cum_rho_arr[0] = 0.0;
    for (int s = 1; s < S; s++) {
        cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double *d_pos, *d_mu, *d_cum_rho, *d_midpoints, *d_prior;
    int *d_pi, *d_pj;
    float* d_gamma;
    double* d_loglik;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cum_rho, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_midpoints, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prior, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gamma, (size_t)S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loglik, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos_ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cum_rho, cum_rho_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midpoints, midpoints.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));

    int hi = pair.first, hj = pair.second;
    CUDA_CHECK(cudaMemcpy(d_pi, &hi, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, &hj, sizeof(int), cudaMemcpyHostToDevice));

    hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                              d_mu, d_cum_rho, d_midpoints, d_prior,
                              nullptr,  // no EP messages
                              d_pi, d_pj, 1,
                              d_gamma, d_loglik,
                              nullptr, nullptr, nullptr, nullptr,
                              K_bins, FULL_GAMMA);

    auto result = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)K_bins});
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_gamma,
                          (size_t)S * K_bins * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_pos);
    cudaFree(d_mu);
    cudaFree(d_cum_rho);
    cudaFree(d_midpoints);
    cudaFree(d_prior);
    cudaFree(d_pi);
    cudaFree(d_pj);
    cudaFree(d_gamma);
    cudaFree(d_loglik);

    return result;
}

// ============================================================
// HMM log-likelihood
// ============================================================
double py_hmm_log_likelihood(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::pair<int, int> pair,
    int K_bins,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    double t_max)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;

    if (K_bins != 32 && K_bins != 64 && K_bins != 128)
        throw std::runtime_error("K must be 32, 64, or 128");

    if (t_max <= 0) t_max = 10.0 * Ne;

    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);

    std::vector<double> mu_arr(S, mu_scalar);
    std::vector<double> cum_rho_arr(S);
    double* pos_ptr = (double*)pos_buf.ptr;
    cum_rho_arr[0] = 0.0;
    for (int s = 1; s < S; s++) {
        cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
    }

    uint8_t* d_G;
    uint64_t* d_packed;
    double *d_pos, *d_mu, *d_cum_rho, *d_midpoints, *d_prior;
    int *d_pi, *d_pj;
    float* d_gamma;
    double* d_loglik;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cum_rho, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_midpoints, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prior, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gamma, (size_t)S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loglik, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos_ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cum_rho, cum_rho_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midpoints, midpoints.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));

    int hi = pair.first, hj = pair.second;
    CUDA_CHECK(cudaMemcpy(d_pi, &hi, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, &hj, sizeof(int), cudaMemcpyHostToDevice));

    hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                              d_mu, d_cum_rho, d_midpoints, d_prior,
                              nullptr, d_pi, d_pj, 1,
                              d_gamma, d_loglik,
                              nullptr, nullptr, nullptr, nullptr,
                              K_bins, FULL_GAMMA);

    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_loglik, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_packed); cudaFree(d_pos); cudaFree(d_mu);
    cudaFree(d_cum_rho); cudaFree(d_midpoints); cudaFree(d_prior);
    cudaFree(d_pi); cudaFree(d_pj); cudaFree(d_gamma); cudaFree(d_loglik);

    return result;
}

// ============================================================
// Time midpoints helper
// ============================================================
py::array_t<double> py_time_midpoints(int K_bins, double Ne, double t_max) {
    if (t_max <= 0) t_max = 10.0 * Ne;
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);
    auto result = py::array_t<double>(K_bins);
    std::memcpy(result.mutable_data(), midpoints.data(), K_bins * sizeof(double));
    return result;
}

py::array_t<double> py_time_boundaries(int K_bins, double Ne, double t_max) {
    if (t_max <= 0) t_max = 10.0 * Ne;
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);
    auto result = py::array_t<double>(K_bins + 1);
    std::memcpy(result.mutable_data(), boundaries.data(), (K_bins + 1) * sizeof(double));
    return result;
}

// ============================================================
// Batched HMM posterior (multiple pairs)
// ============================================================
py::tuple py_hmm_posterior_batched(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    int K_bins,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    double t_max)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    if (K_bins != 32 && K_bins != 64 && K_bins != 128)
        throw std::runtime_error("K must be 32, 64, or 128");

    if (t_max <= 0) t_max = 10.0 * Ne;

    // Coalescent prior
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);

    // Mu and rho arrays
    std::vector<double> mu_arr(S, mu_scalar);
    std::vector<double> cum_rho_arr(S);
    double* pos_ptr = (double*)pos_buf.ptr;
    cum_rho_arr[0] = 0.0;
    for (int s = 1; s < S; s++) {
        cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
    }

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double *d_pos, *d_mu, *d_cum_rho, *d_midpoints, *d_prior;
    int *d_pi, *d_pj;
    float* d_gamma;
    double* d_loglik;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cum_rho, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_midpoints, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prior, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gamma, (size_t)n_pairs * S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loglik, n_pairs * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos_ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cum_rho, cum_rho_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midpoints, midpoints.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate summary buffers
    float *d_mean, *d_lower, *d_upper;
    CUDA_CHECK(cudaMalloc(&d_mean, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lower, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upper, (size_t)n_pairs * S * sizeof(float)));

    // Fused forward-backward + summary extraction (SUMMARY_ONLY mode)
    hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                              d_mu, d_cum_rho, d_midpoints, d_prior,
                              nullptr, d_pi, d_pj, n_pairs,
                              d_gamma, d_loglik,
                              d_mean, d_lower, d_upper, nullptr,
                              K_bins, SUMMARY_ONLY);

    // gamma is scratch in SUMMARY_ONLY mode — skip the massive D2H copy
    auto gamma_out = py::array_t<float>(std::vector<ssize_t>{0, 0, 0});  // empty placeholder
    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto loglik_out = py::array_t<double>((ssize_t)n_pairs);

    CUDA_CHECK(cudaMemcpy(mean_out.mutable_data(), d_mean,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lower_out.mutable_data(), d_lower,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_out.mutable_data(), d_upper,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loglik_out.mutable_data(), d_loglik,
                          n_pairs * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_packed); cudaFree(d_pos); cudaFree(d_mu);
    cudaFree(d_cum_rho); cudaFree(d_midpoints); cudaFree(d_prior);
    cudaFree(d_pi); cudaFree(d_pj); cudaFree(d_gamma); cudaFree(d_loglik);
    cudaFree(d_mean); cudaFree(d_lower); cudaFree(d_upper);

    return py::make_tuple(gamma_out, mean_out, lower_out, upper_out, loglik_out);
}


// ============================================================
// Site pi (nucleotide diversity)
// ============================================================

// Forward declaration (defined in tier1_divergence.cu)
extern void site_pi_gpu(const uint64_t* packed, int n_words, int S,
                        const int* pair_i, const int* pair_j, int n_pairs,
                        float* pi_out);

py::array_t<float> py_site_pi(
    py::array_t<uint8_t, py::array::c_style> G,
    int n_sample_pairs)
{
    auto buf = G.request();
    int n = (int)buf.shape[0];
    int S = (int)buf.shape[1];
    int n_words = (S + 63) / 64;

    // Generate random pairs
    int total_possible = n * (n - 1) / 2;
    int n_pairs = std::min(n_sample_pairs, total_possible);

    std::vector<int> pi_arr(n_pairs), pj_arr(n_pairs);
    // Deterministic pair selection: first n_pairs pairs in canonical order
    int p = 0;
    for (int i = 1; i < n && p < n_pairs; i++) {
        for (int j = 0; j < i && p < n_pairs; j++) {
            pi_arr[p] = i;
            pj_arr[p] = j;
            p++;
        }
    }

    // Bitpack
    uint8_t* d_G;
    uint64_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    // Pair indices
    int *d_pi, *d_pj;
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi_arr.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj_arr.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Compute pi
    float* d_pi_out;
    CUDA_CHECK(cudaMalloc(&d_pi_out, S * sizeof(float)));

    site_pi_gpu(d_packed, n_words, S, d_pi, d_pj, n_pairs, d_pi_out);

    auto result = py::array_t<float>(S);
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_pi_out,
                          S * sizeof(float), cudaMemcpyDeviceToHost));

    // Normalize by n_pairs to get average divergence
    float* ptr = result.mutable_data();
    for (int s = 0; s < S; s++) {
        ptr[s] /= (float)n_pairs;
    }

    cudaFree(d_packed);
    cudaFree(d_pi);
    cudaFree(d_pj);
    cudaFree(d_pi_out);
    return result;
}

// ============================================================
// PELT changepoint detection
// ============================================================
py::dict py_pelt_changepoint(
    py::array_t<int64_t, py::array::c_style> prefix_arr,
    py::array_t<double, py::array::c_style> positions_arr,
    int n_pairs,
    double mu,
    double penalty)
{
    auto prefix_buf = prefix_arr.request();
    auto pos_buf = positions_arr.request();

    if (prefix_buf.ndim != 2)
        throw std::runtime_error("prefix must be 2D (n_pairs x S)");

    int S = (int)prefix_buf.shape[1];
    if (n_pairs != (int)prefix_buf.shape[0])
        throw std::runtime_error("n_pairs must match prefix.shape[0]");

    int max_segments = PELT_MAX_SEGMENTS;

    // Copy prefix to device
    int64_t* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)n_pairs * S * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_prefix, prefix_buf.ptr,
                          (size_t)n_pairs * S * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Copy positions to device
    double* d_positions;
    CUDA_CHECK(cudaMalloc(&d_positions, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_positions, pos_buf.ptr,
                          S * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate output buffers on device
    int* d_n_segments;
    int* d_seg_starts;
    int* d_seg_ends;
    float* d_seg_tmrca;
    int* d_seg_counts;

    CUDA_CHECK(cudaMalloc(&d_n_segments, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_starts, (size_t)n_pairs * max_segments * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_ends, (size_t)n_pairs * max_segments * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_tmrca, (size_t)n_pairs * max_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seg_counts, (size_t)n_pairs * max_segments * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_n_segments, 0, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_seg_starts, 0, (size_t)n_pairs * max_segments * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_seg_ends, 0, (size_t)n_pairs * max_segments * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_seg_tmrca, 0, (size_t)n_pairs * max_segments * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_seg_counts, 0, (size_t)n_pairs * max_segments * sizeof(int)));

    // Run PELT on GPU
    pelt_changepoint_gpu(
        d_prefix, d_positions, S, n_pairs,
        mu, penalty,
        d_n_segments, d_seg_starts, d_seg_ends,
        d_seg_tmrca, d_seg_counts, max_segments
    );

    // Copy results back to host
    auto n_segments_out = py::array_t<int>((ssize_t)n_pairs);
    auto seg_starts_out = py::array_t<int>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)max_segments});
    auto seg_ends_out = py::array_t<int>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)max_segments});
    auto seg_tmrca_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)max_segments});
    auto seg_counts_out = py::array_t<int>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)max_segments});

    CUDA_CHECK(cudaMemcpy(n_segments_out.mutable_data(), d_n_segments,
                          n_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seg_starts_out.mutable_data(), d_seg_starts,
                          (size_t)n_pairs * max_segments * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seg_ends_out.mutable_data(), d_seg_ends,
                          (size_t)n_pairs * max_segments * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seg_tmrca_out.mutable_data(), d_seg_tmrca,
                          (size_t)n_pairs * max_segments * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seg_counts_out.mutable_data(), d_seg_counts,
                          (size_t)n_pairs * max_segments * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_prefix);
    cudaFree(d_positions);
    cudaFree(d_n_segments);
    cudaFree(d_seg_starts);
    cudaFree(d_seg_ends);
    cudaFree(d_seg_tmrca);
    cudaFree(d_seg_counts);

    // Build result dict
    py::dict result;
    result["n_segments"] = n_segments_out;
    result["seg_starts"] = seg_starts_out;
    result["seg_ends"] = seg_ends_out;
    result["seg_tmrca"] = seg_tmrca_out;
    result["seg_counts"] = seg_counts_out;
    return result;
}

// ============================================================
// EP inference: HMM + ultrametric loop, all on GPU
// ============================================================
py::dict py_ep_infer(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    int m_haplotypes,    // number of haplotypes in subsample (for ultrametric)
    int K_bins,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    double t_max,
    int max_iterations,
    double damping,
    double convergence_tol)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();
    int m = m_haplotypes;
    int n_sub_pairs = m * (m - 1) / 2;

    if (K_bins != 32 && K_bins != 64 && K_bins != 128)
        throw std::runtime_error("K must be 32, 64, or 128");
    if (t_max <= 0) t_max = 10.0 * Ne;

    // Coalescent prior
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);

    // Mu and rho arrays
    std::vector<double> mu_arr(S, mu_scalar);
    std::vector<double> cum_rho_arr(S);
    double* pos_ptr = (double*)pos_buf.ptr;
    cum_rho_arr[0] = 0.0;
    for (int s = 1; s < S; s++) {
        cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
    }

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // Build mapping: for each pair in the pairs list, what is its index
    // in the subsample's pair list? We need this to scatter/gather between
    // the HMM (indexed by pairs list) and ultrametric (indexed by subsample pairs).
    //
    // The ultrametric kernel uses pair_index(i, j) for i,j in [0, m).
    // But the HMM pairs reference haplotype indices in [0, n).
    // We need to map: subsample haplotype ID -> local ID in [0, m).
    //
    // Collect the unique haplotypes used in pairs (should be exactly m).
    std::vector<int> hap_set;
    for (auto& pr : pairs) {
        hap_set.push_back(pr.first);
        hap_set.push_back(pr.second);
    }
    std::sort(hap_set.begin(), hap_set.end());
    hap_set.erase(std::unique(hap_set.begin(), hap_set.end()), hap_set.end());
    if ((int)hap_set.size() != m) {
        throw std::runtime_error(
            "Number of unique haplotypes in pairs (" + std::to_string(hap_set.size()) +
            ") must equal m_haplotypes (" + std::to_string(m) + ")");
    }

    // hap_to_local: global hap ID -> local [0, m)
    std::vector<int> hap_to_local(n, -1);
    for (int i = 0; i < m; i++) {
        hap_to_local[hap_set[i]] = i;
    }

    // Build mapping: pairs list index -> subsample pair index
    // subsample pair index = local_i * (local_i - 1) / 2 + local_j  (local_i > local_j)
    std::vector<int> pair_to_sub(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        int li = hap_to_local[pi[p]];
        int lj = hap_to_local[pj[p]];
        if (li < lj) std::swap(li, lj);
        pair_to_sub[p] = li * (li - 1) / 2 + lj;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double *d_pos, *d_mu, *d_cum_rho, *d_midpoints, *d_prior;
    int *d_pi, *d_pj;
    float *d_gamma, *d_messages, *d_messages_prev;
    double* d_loglik;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cum_rho, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_midpoints, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prior, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));

    // HMM gamma: [n_pairs × S × K]
    CUDA_CHECK(cudaMalloc(&d_gamma, (size_t)n_pairs * S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loglik, n_pairs * sizeof(double)));

    // Messages: [n_pairs × S × K] — per-pair per-site messages for HMM
    CUDA_CHECK(cudaMalloc(&d_messages, (size_t)n_pairs * S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_messages_prev, (size_t)n_pairs * S * K_bins * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos_ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cum_rho, cum_rho_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midpoints, midpoints.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize messages to 1.0 (neutral multiplicative factor)
    {
        std::vector<float> init_msg((size_t)n_pairs * S * K_bins, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_messages, init_msg.data(),
                              (size_t)n_pairs * S * K_bins * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Ultrametric workspace: per-site view
    // gamma_site: [S × n_sub_pairs × K]
    // messages_site: [S × n_sub_pairs × K]
    float *d_gamma_site, *d_msg_site;
    int* d_assigned;
    CUDA_CHECK(cudaMalloc(&d_gamma_site, (size_t)S * n_sub_pairs * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_msg_site, (size_t)S * n_sub_pairs * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assigned, (size_t)S * n_sub_pairs * sizeof(int)));

    // Convergence check
    float* d_max_delta;
    CUDA_CHECK(cudaMalloc(&d_max_delta, sizeof(float)));

    // Copy pair_to_sub mapping to device for scatter/gather kernels
    int* d_pair_to_sub;
    CUDA_CHECK(cudaMalloc(&d_pair_to_sub, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_to_sub, pair_to_sub.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    // Track per-iteration log-likelihoods
    std::vector<double> iter_logliks;
    int n_iter = 0;
    bool converged = false;

    for (int iter = 0; iter < max_iterations; iter++) {
        n_iter = iter + 1;

        // Save previous messages for convergence check
        CUDA_CHECK(cudaMemcpy(d_messages_prev, d_messages,
                              (size_t)n_pairs * S * K_bins * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // ── Step 1: HMM forward-backward with messages ──
        hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                                  d_mu, d_cum_rho, d_midpoints, d_prior,
                                  d_messages, d_pi, d_pj, n_pairs,
                                  d_gamma, d_loglik,
                                  nullptr, nullptr, nullptr, nullptr,
                                  K_bins, FULL_GAMMA);

        // Collect total log-likelihood
        std::vector<double> h_loglik(n_pairs);
        CUDA_CHECK(cudaMemcpy(h_loglik.data(), d_loglik,
                              n_pairs * sizeof(double), cudaMemcpyDeviceToHost));
        double total_ll = 0;
        for (int p = 0; p < n_pairs; p++) total_ll += h_loglik[p];
        iter_logliks.push_back(total_ll);

        if (iter == max_iterations - 1) break;  // last iteration, skip ultrametric

        // ── Step 2: Scatter gamma to per-site layout for ultrametric ──
        // gamma is [n_pairs × S × K], need gamma_site [S × n_sub_pairs × K]
        // We do this on host for correctness (could be a kernel later)
        {
            size_t gamma_bytes = (size_t)n_pairs * S * K_bins * sizeof(float);
            std::vector<float> h_gamma(n_pairs * (size_t)S * K_bins);
            CUDA_CHECK(cudaMemcpy(h_gamma.data(), d_gamma, gamma_bytes, cudaMemcpyDeviceToHost));

            std::vector<float> h_gamma_site((size_t)S * n_sub_pairs * K_bins, 0.0f);
            for (int p = 0; p < n_pairs; p++) {
                int sub_p = pair_to_sub[p];
                for (int s = 0; s < S; s++) {
                    for (int k = 0; k < K_bins; k++) {
                        h_gamma_site[(size_t)s * n_sub_pairs * K_bins + sub_p * K_bins + k]
                            = h_gamma[(size_t)p * S * K_bins + s * K_bins + k];
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(d_gamma_site, h_gamma_site.data(),
                                  (size_t)S * n_sub_pairs * K_bins * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        // ── Step 3: Ultrametric projection ──
        ultrametric_project_gpu(d_gamma_site, m, K_bins, d_prior, damping,
                                d_msg_site, d_assigned, S, n_sub_pairs);

        // ── Step 4: Gather messages back to per-pair layout ──
        // Convert ultrametric distribution messages to multiplicative EP factors:
        //   ep_msg[k] = ultrametric_msg[k] / prior[k]
        // This way, when the HMM multiplies by ep_msg, the effective per-site
        // prior becomes the ultrametric-projected distribution.
        {
            size_t msg_site_bytes = (size_t)S * n_sub_pairs * K_bins * sizeof(float);
            std::vector<float> h_msg_site(S * (size_t)n_sub_pairs * K_bins);
            CUDA_CHECK(cudaMemcpy(h_msg_site.data(), d_msg_site, msg_site_bytes, cudaMemcpyDeviceToHost));

            std::vector<float> h_messages(n_pairs * (size_t)S * K_bins);
            for (int p = 0; p < n_pairs; p++) {
                int sub_p = pair_to_sub[p];
                for (int s = 0; s < S; s++) {
                    for (int k = 0; k < K_bins; k++) {
                        float um_msg = h_msg_site[(size_t)s * n_sub_pairs * K_bins + sub_p * K_bins + k];
                        float q_k = (float)prior[k];
                        // EP ratio: msg / prior, clamped to avoid numerical blow-up
                        float ep_factor;
                        if (q_k > 1e-10f) {
                            ep_factor = um_msg / q_k;
                        } else {
                            ep_factor = 1.0f;
                        }
                        // Tight clamp to keep messages gentle
                        if (ep_factor > 10.0f) ep_factor = 10.0f;
                        if (ep_factor < 0.1f) ep_factor = 0.1f;
                        h_messages[(size_t)p * S * K_bins + s * K_bins + k] = ep_factor;
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(d_messages, h_messages.data(),
                                  (size_t)n_pairs * S * K_bins * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        // ── Step 5: Convergence check ──
        check_convergence_gpu(d_messages, d_messages_prev,
                              n_pairs * S * K_bins, d_max_delta);
        float h_max_delta;
        CUDA_CHECK(cudaMemcpy(&h_max_delta, d_max_delta, sizeof(float), cudaMemcpyDeviceToHost));

        if (h_max_delta < (float)convergence_tol) {
            converged = true;
            break;
        }
    }

    // ── Extract final summaries ──
    float *d_mean, *d_lower, *d_upper;
    CUDA_CHECK(cudaMalloc(&d_mean, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lower, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upper, (size_t)n_pairs * S * sizeof(float)));

    extract_summaries_gpu(d_gamma, n_pairs, S, d_midpoints,
                          d_mean, d_lower, d_upper, K_bins);

    auto gamma_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S, (ssize_t)K_bins});
    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto loglik_out = py::array_t<double>((ssize_t)n_pairs);

    CUDA_CHECK(cudaMemcpy(gamma_out.mutable_data(), d_gamma,
                          (size_t)n_pairs * S * K_bins * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mean_out.mutable_data(), d_mean,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lower_out.mutable_data(), d_lower,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_out.mutable_data(), d_upper,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loglik_out.mutable_data(), d_loglik,
                          n_pairs * sizeof(double), cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_packed); cudaFree(d_pos); cudaFree(d_mu);
    cudaFree(d_cum_rho); cudaFree(d_midpoints); cudaFree(d_prior);
    cudaFree(d_pi); cudaFree(d_pj); cudaFree(d_gamma); cudaFree(d_loglik);
    cudaFree(d_messages); cudaFree(d_messages_prev);
    cudaFree(d_gamma_site); cudaFree(d_msg_site); cudaFree(d_assigned);
    cudaFree(d_max_delta); cudaFree(d_pair_to_sub);
    cudaFree(d_mean); cudaFree(d_lower); cudaFree(d_upper);

    // Build result
    py::dict result;
    result["gamma"] = gamma_out;
    result["mean"] = mean_out;
    result["lower"] = lower_out;
    result["upper"] = upper_out;
    result["log_likelihood"] = loglik_out;
    result["converged"] = converged;
    result["n_iterations"] = n_iter;

    // Per-iteration log-likelihoods
    auto ll_history = py::array_t<double>(n_iter);
    std::memcpy(ll_history.mutable_data(), iter_logliks.data(), n_iter * sizeof(double));
    result["ll_history"] = ll_history;

    return result;
}

// ============================================================
// Adaptive prior inference: divergence-based prior estimation
//
// Step 1: Compute per-pair genome-wide divergence on GPU
// Step 2: Convert to TMRCA estimates, soft-bin into time bins
// Step 3: (Optional) Refine with EM iterations
// Step 4: Run final HMM with the estimated prior
// ============================================================
py::dict py_adaptive_prior_infer(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    int K_bins,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    double t_max,
    int max_iterations,
    double blend_alpha,
    double convergence_tol)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    if (K_bins != 32 && K_bins != 64 && K_bins != 128)
        throw std::runtime_error("K must be 32, 64, or 128");
    if (t_max <= 0) t_max = 10.0 * Ne;

    // Coalescent prior (initial, constant-Ne assumption)
    std::vector<double> boundaries(K_bins + 1);
    std::vector<double> midpoints(K_bins);
    std::vector<double> prior(K_bins);
    compute_coalescent_prior(Ne, t_max, boundaries.data(), midpoints.data(), prior.data(), K_bins);

    // Mu and rho arrays
    std::vector<double> mu_arr(S, mu_scalar);
    std::vector<double> cum_rho_arr(S);
    double* pos_ptr = (double*)pos_buf.ptr;
    cum_rho_arr[0] = 0.0;
    for (int s = 1; s < S; s++) {
        cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
    }

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // ══════════════════════════════════════════════════════════════
    // GPU allocations
    // ══════════════════════════════════════════════════════════════
    uint8_t* d_G;
    uint64_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    int *d_pi, *d_pj;
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    double *d_pos, *d_mu, *d_cum_rho, *d_midpoints, *d_prior;
    float* d_gamma;
    double* d_loglik;
    double* d_q_emp;

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cum_rho, S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_midpoints, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prior, K_bins * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gamma, (size_t)n_pairs * S * K_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loglik, n_pairs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q_emp, K_bins * sizeof(double)));

    // Summary buffers (allocated once, reused for final run)
    float *d_mean, *d_lower, *d_upper;
    CUDA_CHECK(cudaMalloc(&d_mean, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lower, (size_t)n_pairs * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upper, (size_t)n_pairs * S * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos, pos_ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cum_rho, cum_rho_arr.data(), S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midpoints, midpoints.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));

    // ══════════════════════════════════════════════════════════════
    // HMM-based EM: iteratively update coalescent prior from
    // per-site posteriors γ(s,k). Uses SUMMARY_AND_EM mode to fuse
    // forward-backward + posterior aggregation in a single kernel.
    //
    // E-step: run forward-backward → γ_p(s,k) for each pair & site
    // M-step: q_new[k] = (1/PS) Σ_p Σ_s γ_p(s,k) (accumulated via atomicAdd)
    //
    // We use geometric-mean damping for stability:
    //   prior_new[k] ∝ prior[k]^(1-α) * q_empirical[k]^α
    // ══════════════════════════════════════════════════════════════
    std::vector<double> iter_logliks;
    std::vector<std::vector<double>> prior_history;
    prior_history.push_back(prior);
    int n_iter = 0;
    bool converged = false;
    double inv_count = 1.0 / ((double)n_pairs * S);
    double em_alpha = blend_alpha;  // EM step size (default 0.7)

    for (int iter = 0; iter < max_iterations; iter++) {
        n_iter = iter + 1;

        bool is_last = (iter == max_iterations - 1);
        bool need_em = !is_last;

        // Zero q accumulator for EM
        if (need_em) {
            CUDA_CHECK(cudaMemset(d_q_emp, 0, K_bins * sizeof(double)));
        }

        // E-step: run HMM with current prior
        // Use SUMMARY_AND_EM for intermediate iterations (need q_accum),
        // SUMMARY_ONLY for the final iteration (just need summaries)
        hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                                  d_mu, d_cum_rho, d_midpoints, d_prior,
                                  nullptr, d_pi, d_pj, n_pairs,
                                  d_gamma, d_loglik,
                                  d_mean, d_lower, d_upper,
                                  need_em ? d_q_emp : nullptr,
                                  K_bins, need_em ? SUMMARY_AND_EM : SUMMARY_ONLY);

        // Collect log-likelihood
        std::vector<double> h_loglik(n_pairs);
        CUDA_CHECK(cudaMemcpy(h_loglik.data(), d_loglik,
                              n_pairs * sizeof(double), cudaMemcpyDeviceToHost));
        double total_ll = 0;
        for (int p = 0; p < n_pairs; p++) total_ll += h_loglik[p];
        iter_logliks.push_back(total_ll);

        if (is_last) break;

        // M-step: read fused q accumulator (no separate aggregate_posteriors call)
        std::vector<double> q_emp(K_bins);
        CUDA_CHECK(cudaMemcpy(q_emp.data(), d_q_emp, K_bins * sizeof(double), cudaMemcpyDeviceToHost));

        // Geometric mean update: prior_new ∝ prior^(1-α) * q_emp^α
        double total = 0.0;
        for (int k = 0; k < K_bins; k++) {
            double emp_k = q_emp[k] * inv_count;
            if (emp_k < 1e-30) emp_k = 1e-30;
            if (prior[k] < 1e-30) prior[k] = 1e-30;
            prior[k] = exp((1.0 - em_alpha) * log(prior[k]) + em_alpha * log(emp_k));
            total += prior[k];
        }
        if (total > 0.0) {
            for (int k = 0; k < K_bins; k++) prior[k] /= total;
        }

        prior_history.push_back(prior);

        // Convergence check
        double max_delta = 0.0;
        const auto& prev = prior_history[prior_history.size() - 2];
        for (int k = 0; k < K_bins; k++) {
            double d = std::abs(prior[k] - prev[k]);
            if (d > max_delta) max_delta = d;
        }
        if (max_delta < convergence_tol) {
            converged = true;
            // Final run with converged prior (SUMMARY_ONLY for output)
            CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
            hmm_forward_backward_gpu(d_packed, n_words, d_pos, S,
                                      d_mu, d_cum_rho, d_midpoints, d_prior,
                                      nullptr, d_pi, d_pj, n_pairs,
                                      d_gamma, d_loglik,
                                      d_mean, d_lower, d_upper, nullptr,
                                      K_bins, SUMMARY_ONLY);
            CUDA_CHECK(cudaMemcpy(h_loglik.data(), d_loglik,
                                  n_pairs * sizeof(double), cudaMemcpyDeviceToHost));
            total_ll = 0;
            for (int p = 0; p < n_pairs; p++) total_ll += h_loglik[p];
            iter_logliks.push_back(total_ll);
            n_iter++;
            break;
        }

        CUDA_CHECK(cudaMemcpy(d_prior, prior.data(), K_bins * sizeof(double), cudaMemcpyHostToDevice));
    }

    // gamma is scratch in fused modes — skip the massive D2H copy
    auto gamma_out = py::array_t<float>(std::vector<ssize_t>{0, 0, 0});  // empty placeholder
    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S});
    auto loglik_out = py::array_t<double>((ssize_t)n_pairs);

    CUDA_CHECK(cudaMemcpy(mean_out.mutable_data(), d_mean,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lower_out.mutable_data(), d_lower,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_out.mutable_data(), d_upper,
                          (size_t)n_pairs * S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(loglik_out.mutable_data(), d_loglik,
                          n_pairs * sizeof(double), cudaMemcpyDeviceToHost));

    // Final prior
    auto prior_out = py::array_t<double>(K_bins);
    std::memcpy(prior_out.mutable_data(), prior.data(), K_bins * sizeof(double));

    // Clean up
    cudaFree(d_packed); cudaFree(d_pos); cudaFree(d_mu);
    cudaFree(d_cum_rho); cudaFree(d_midpoints); cudaFree(d_prior);
    cudaFree(d_pi); cudaFree(d_pj); cudaFree(d_gamma); cudaFree(d_loglik);
    cudaFree(d_q_emp); cudaFree(d_mean); cudaFree(d_lower); cudaFree(d_upper);

    // Build result
    py::dict result;
    result["gamma"] = gamma_out;
    result["mean"] = mean_out;
    result["lower"] = lower_out;
    result["upper"] = upper_out;
    result["log_likelihood"] = loglik_out;
    result["prior"] = prior_out;
    result["converged"] = converged;
    result["n_iterations"] = n_iter;

    auto ll_history = py::array_t<double>(n_iter);
    std::memcpy(ll_history.mutable_data(), iter_logliks.data(), n_iter * sizeof(double));
    result["ll_history"] = ll_history;

    return result;
}

// ============================================================
// Persistent GPU context for batched HMM inference
// Holds packed genotypes + parameter arrays on GPU, avoids
// re-uploading per batch. Auto-chunks pairs to fit in VRAM.
// ============================================================
class HMMContext {
    // Persistent GPU buffers
    uint64_t* d_packed_ = nullptr;
    double* d_pos_ = nullptr;
    double* d_mu_ = nullptr;
    double* d_cum_rho_ = nullptr;
    double* d_midpoints_ = nullptr;
    double* d_prior_ = nullptr;

    // Per-batch scratch (allocated to max_batch_)
    float* d_gamma_ = nullptr;      // alpha scratch
    double* d_loglik_ = nullptr;
    float* d_mean_ = nullptr;
    float* d_lower_ = nullptr;
    float* d_upper_ = nullptr;
    double* d_q_accum_ = nullptr;
    int* d_pi_ = nullptr;
    int* d_pj_ = nullptr;

    int n_haps_, n_words_, S_, K_;
    int max_batch_;               // current scratch allocation size

    // Host-side copies for prior updates
    std::vector<double> midpoints_h_;
    std::vector<double> prior_h_;

    void alloc_scratch(int batch_size) {
        if (batch_size <= max_batch_) return;
        free_scratch();
        max_batch_ = batch_size;
        CUDA_CHECK(cudaMalloc(&d_gamma_, (size_t)max_batch_ * S_ * K_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_loglik_, max_batch_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mean_, (size_t)max_batch_ * S_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_lower_, (size_t)max_batch_ * S_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upper_, (size_t)max_batch_ * S_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q_accum_, K_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pi_, max_batch_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pj_, max_batch_ * sizeof(int)));
    }

    void free_scratch() {
        if (d_gamma_) { cudaFree(d_gamma_); d_gamma_ = nullptr; }
        if (d_loglik_) { cudaFree(d_loglik_); d_loglik_ = nullptr; }
        if (d_mean_) { cudaFree(d_mean_); d_mean_ = nullptr; }
        if (d_lower_) { cudaFree(d_lower_); d_lower_ = nullptr; }
        if (d_upper_) { cudaFree(d_upper_); d_upper_ = nullptr; }
        if (d_q_accum_) { cudaFree(d_q_accum_); d_q_accum_ = nullptr; }
        if (d_pi_) { cudaFree(d_pi_); d_pi_ = nullptr; }
        if (d_pj_) { cudaFree(d_pj_); d_pj_ = nullptr; }
        max_batch_ = 0;
    }

    int compute_max_batch() const {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        // Reserve 512MB headroom
        if (free_mem < 512ULL * 1024 * 1024) return 1;
        free_mem -= 512ULL * 1024 * 1024;
        // Per-pair memory: gamma scratch + summaries + loglik + pair indices
        size_t per_pair = (size_t)S_ * K_ * sizeof(float)     // d_gamma
                        + (size_t)S_ * sizeof(float) * 3      // d_mean, d_lower, d_upper
                        + sizeof(double)                       // d_loglik
                        + sizeof(int) * 2;                     // d_pi, d_pj
        int max_batch = (int)(free_mem / per_pair);
        return std::max(max_batch, 1);
    }

public:
    HMMContext(
        py::array_t<uint8_t, py::array::c_style> G,
        py::array_t<double, py::array::c_style> positions_arr,
        int K_bins,
        double Ne,
        double mu_scalar,
        double rho_scalar,
        double t_max)
    {
        auto g_buf = G.request();
        auto pos_buf = positions_arr.request();

        n_haps_ = (int)g_buf.shape[0];
        S_ = (int)g_buf.shape[1];
        n_words_ = (S_ + 63) / 64;
        K_ = K_bins;

        if (K_ != 32 && K_ != 64 && K_ != 128)
            throw std::runtime_error("K must be 32, 64, or 128");
        if (t_max <= 0) t_max = 10.0 * Ne;

        // Coalescent prior
        std::vector<double> boundaries(K_ + 1);
        midpoints_h_.resize(K_);
        prior_h_.resize(K_);
        compute_coalescent_prior(Ne, t_max, boundaries.data(),
                                 midpoints_h_.data(), prior_h_.data(), K_);

        // Mu and rho arrays (uniform)
        std::vector<double> mu_arr(S_, mu_scalar);
        std::vector<double> cum_rho_arr(S_);
        double* pos_ptr = (double*)pos_buf.ptr;
        cum_rho_arr[0] = 0.0;
        for (int s = 1; s < S_; s++) {
            cum_rho_arr[s] = cum_rho_arr[s - 1] + rho_scalar * (pos_ptr[s] - pos_ptr[s - 1]);
        }


        // Upload and bitpack genotypes
        uint8_t* d_G;
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)n_haps_ * S_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_packed_, (size_t)n_haps_ * n_words_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_packed_, 0, (size_t)n_haps_ * n_words_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n_haps_ * S_, cudaMemcpyHostToDevice));
        bitpack_genotypes_gpu(d_G, d_packed_, n_haps_, S_, n_words_);
        cudaFree(d_G);

        // Upload parameter arrays
        CUDA_CHECK(cudaMalloc(&d_pos_, S_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu_, S_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cum_rho_, S_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_midpoints_, K_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_prior_, K_ * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_pos_, pos_ptr, S_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mu_, mu_arr.data(), S_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cum_rho_, cum_rho_arr.data(), S_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_midpoints_, midpoints_h_.data(), K_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prior_, prior_h_.data(), K_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~HMMContext() {
        free_scratch();
        if (d_packed_) cudaFree(d_packed_);
        if (d_pos_) cudaFree(d_pos_);
        if (d_mu_) cudaFree(d_mu_);
        if (d_cum_rho_) cudaFree(d_cum_rho_);
        if (d_midpoints_) cudaFree(d_midpoints_);
        if (d_prior_) cudaFree(d_prior_);
    }

    // Non-copyable
    HMMContext(const HMMContext&) = delete;
    HMMContext& operator=(const HMMContext&) = delete;



    int n_haps() const { return n_haps_; }
    int n_sites() const { return S_; }
    int n_bins() const { return K_; }

    py::tuple run_batch(std::vector<std::pair<int, int>> pairs) {
        int n_pairs = (int)pairs.size();
        if (n_pairs == 0) {
            return py::make_tuple(
                py::array_t<float>(std::vector<ssize_t>{0, (ssize_t)S_}),
                py::array_t<float>(std::vector<ssize_t>{0, (ssize_t)S_}),
                py::array_t<float>(std::vector<ssize_t>{0, (ssize_t)S_}),
                py::array_t<double>(0));
        }

        // Determine chunk size
        int max_chunk = compute_max_batch();

        // Output arrays
        auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S_});
        auto lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S_});
        auto upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)n_pairs, (ssize_t)S_});
        auto loglik_out = py::array_t<double>((ssize_t)n_pairs);

        float* h_mean = mean_out.mutable_data();
        float* h_lower = lower_out.mutable_data();
        float* h_upper = upper_out.mutable_data();
        double* h_loglik = loglik_out.mutable_data();

        // Process in chunks
        for (int offset = 0; offset < n_pairs; offset += max_chunk) {
            int chunk = std::min(max_chunk, n_pairs - offset);

            // Ensure scratch is big enough
            alloc_scratch(chunk);

            // Upload pair indices
            std::vector<int> pi(chunk), pj(chunk);
            for (int i = 0; i < chunk; i++) {
                pi[i] = pairs[offset + i].first;
                pj[i] = pairs[offset + i].second;
            }
            CUDA_CHECK(cudaMemcpy(d_pi_, pi.data(), chunk * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pj_, pj.data(), chunk * sizeof(int), cudaMemcpyHostToDevice));

            // Run kernel (SUMMARY_ONLY)
            hmm_forward_backward_gpu(
                d_packed_, n_words_, d_pos_, S_,
                d_mu_, d_cum_rho_, d_midpoints_, d_prior_,
                nullptr, d_pi_, d_pj_, chunk,
                d_gamma_, d_loglik_,
                d_mean_, d_lower_, d_upper_, nullptr,
                K_, SUMMARY_ONLY);

            // Download results for this chunk
            size_t chunk_sites = (size_t)chunk * S_;
            CUDA_CHECK(cudaMemcpy(h_mean + (size_t)offset * S_, d_mean_,
                                  chunk_sites * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lower + (size_t)offset * S_, d_lower_,
                                  chunk_sites * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_upper + (size_t)offset * S_, d_upper_,
                                  chunk_sites * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_loglik + offset, d_loglik_,
                                  chunk * sizeof(double), cudaMemcpyDeviceToHost));
        }

        return py::make_tuple(mean_out, lower_out, upper_out, loglik_out);
    }
};

// ============================================================
// Gamma-SMC forward filtering
// ============================================================
py::dict py_gamma_smc_forward(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    int stride,
    bool mean_only)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();
    int out_S = (stride == 1) ? S : (S + stride - 1) / stride;

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double* d_pos;
    int *d_pi, *d_pj;
    float *d_mean, *d_lower, *d_upper;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Determine chunk size based on available VRAM (70% of free)
    int n_arrays = mean_only ? 1 : 3;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t output_per_pair = (size_t)out_S * n_arrays * sizeof(float);
    size_t usable = (size_t)(free_mem * 0.7);
    int chunk_pairs = n_pairs;
    if (output_per_pair > 0 && (size_t)n_pairs * output_per_pair > usable) {
        chunk_pairs = std::max(1, (int)(usable / output_per_pair));
    }

    // Allocate GPU output buffers for one chunk
    size_t chunk_out_size = (size_t)chunk_pairs * out_S;
    d_lower = nullptr;
    d_upper = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mean, chunk_out_size * sizeof(float)));
    if (!mean_only) {
        CUDA_CHECK(cudaMalloc(&d_lower, chunk_out_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upper, chunk_out_size * sizeof(float)));
    }

    // Host output: site-major [out_S × n_pairs], C-contiguous.
    // User can .T for [n_pairs × out_S] view (free, no copy).
    // Using [out_S, n_pairs] shape matches GPU layout → direct memcpy, no scatter.
    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)out_S, (ssize_t)n_pairs});
    py::array_t<float> lower_out, upper_out;
    if (!mean_only) {
        lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)out_S, (ssize_t)n_pairs});
        upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)out_S, (ssize_t)n_pairs});
    }

    float* h_mean = mean_out.mutable_data();
    float* h_lower = mean_only ? nullptr : lower_out.mutable_data();
    float* h_upper = mean_only ? nullptr : upper_out.mutable_data();

    if (chunk_pairs >= n_pairs) {
        // Fast path: no chunking — single kernel launch + single memcpy
        gamma_smc_forward_gpu(d_packed, n_words, d_pos, S,
                              (float)mu_scalar, (float)rho_scalar, (float)Ne,
                              d_pi, d_pj, n_pairs,
                              d_mean, d_lower, d_upper, stride);

        size_t total_bytes = (size_t)out_S * n_pairs * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_mean, d_mean, total_bytes, cudaMemcpyDeviceToHost));
        if (!mean_only) {
            CUDA_CHECK(cudaMemcpy(h_lower, d_lower, total_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_upper, d_upper, total_bytes, cudaMemcpyDeviceToHost));
        }
    } else {
        // Chunked path: kernel writes [out_S × chunk], D2H into correct
        // position in host array. Since chunks slice along pair dimension,
        // the site-major layout means chunk data for site s is at
        // offset s*chunk in GPU buffer, and goes to s*n_pairs+offset in host.
        for (int offset = 0; offset < n_pairs; offset += chunk_pairs) {
            int chunk = std::min(chunk_pairs, n_pairs - offset);

            gamma_smc_forward_gpu(d_packed, n_words, d_pos, S,
                                  (float)mu_scalar, (float)rho_scalar, (float)Ne,
                                  d_pi + offset, d_pj + offset, chunk,
                                  d_mean, d_lower, d_upper, stride);

            // Use cudaMemcpy2D: copy [out_S × chunk] into strided host buffer
            // src: contiguous [out_S × chunk], pitch = chunk * sizeof(float)
            // dst: [out_S × n_pairs] at column offset, pitch = n_pairs * sizeof(float)
            auto copy2d = [&](float* d_src, float* h_dst) {
                CUDA_CHECK(cudaMemcpy2D(
                    h_dst + offset,                     // dst ptr (offset by pair)
                    (size_t)n_pairs * sizeof(float),    // dst pitch
                    d_src,                              // src ptr
                    (size_t)chunk * sizeof(float),      // src pitch
                    (size_t)chunk * sizeof(float),      // width (bytes per row)
                    out_S,                              // height (number of rows/sites)
                    cudaMemcpyDeviceToHost));
            };
            copy2d(d_mean, h_mean);
            if (!mean_only) {
                copy2d(d_lower, h_lower);
                copy2d(d_upper, h_upper);
            }
        }
    }

    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_mean);
    if (d_lower) cudaFree(d_lower);
    if (d_upper) cudaFree(d_upper);

    py::dict result;
    result["mean"] = mean_out;
    if (!mean_only) {
        result["lower"] = lower_out;
        result["upper"] = upper_out;
    }
    return result;
}

// ============================================================
// Gamma-SMC quantized forward filtering
// ============================================================
py::dict py_gamma_smc_forward_quantized(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    int stride,
    int bits)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();
    int out_S = (stride == 1) ? S : (S + stride - 1) / stride;

    if (bits != 4 && bits != 8)
        throw std::runtime_error("bits must be 4 or 8");

    // Log-scale range based on Ne: [1 generation, 20*Ne generations]
    float log_min = 0.0f;                      // log(1)
    float log_max = logf(20.0f * (float)Ne);   // log(20*Ne)

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double* d_pos;
    int *d_pi, *d_pj;
    unsigned char* d_q;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Output size depends on bits
    int out_bytes_per_pair;
    if (bits == 8) {
        out_bytes_per_pair = out_S;
    } else {
        out_bytes_per_pair = (out_S + 1) / 2;  // ceil(out_S / 2)
    }
    size_t q_total = (size_t)out_bytes_per_pair * n_pairs;

    // Chunk for VRAM
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t usable = (size_t)(free_mem * 0.7);
    int chunk_pairs = n_pairs;
    if ((size_t)n_pairs * out_bytes_per_pair > usable) {
        chunk_pairs = std::max(1, (int)(usable / out_bytes_per_pair));
    }

    size_t chunk_q_size = (size_t)chunk_pairs * out_bytes_per_pair;
    CUDA_CHECK(cudaMalloc(&d_q, chunk_q_size));

    // Host output
    int q_height = (bits == 8) ? out_S : (out_S + 1) / 2;
    auto q_out = py::array_t<uint8_t>(std::vector<ssize_t>{
        (ssize_t)q_height, (ssize_t)n_pairs});
    uint8_t* h_q = (uint8_t*)q_out.mutable_data();

    if (chunk_pairs >= n_pairs) {
        gamma_smc_forward_quantized_gpu(d_packed, n_words, d_pos, S,
                                         (float)mu_scalar, (float)rho_scalar, (float)Ne,
                                         d_pi, d_pj, n_pairs,
                                         d_q, log_min, log_max, stride, bits);
        CUDA_CHECK(cudaMemcpy(h_q, d_q, q_total, cudaMemcpyDeviceToHost));
    } else {
        for (int offset = 0; offset < n_pairs; offset += chunk_pairs) {
            int chunk = std::min(chunk_pairs, n_pairs - offset);

            gamma_smc_forward_quantized_gpu(d_packed, n_words, d_pos, S,
                                             (float)mu_scalar, (float)rho_scalar, (float)Ne,
                                             d_pi + offset, d_pj + offset, chunk,
                                             d_q, log_min, log_max, stride, bits);

            CUDA_CHECK(cudaMemcpy2D(
                h_q + offset,
                (size_t)n_pairs,
                d_q,
                (size_t)chunk,
                (size_t)chunk,
                q_height,
                cudaMemcpyDeviceToHost));
        }
    }

    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_q);

    py::dict result;
    result["quantized"] = q_out;
    result["bits"] = bits;
    result["log_min"] = log_min;
    result["log_max"] = log_max;
    result["out_S"] = out_S;
    result["n_pairs"] = n_pairs;
    return result;
}

// ============================================================

// ============================================================
// Gamma-SMC site summary (GPU-side reduction, minimal D2H)
// ============================================================
py::dict py_gamma_smc_site_summary(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne, double mu_scalar, double rho_scalar,
    int stride)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();
    int out_S = (stride == 1) ? S : (S + stride - 1) / stride;

    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    uint8_t* d_G; uint64_t* d_packed; double* d_pos;
    int *d_pi, *d_pj;
    float *d_site_mean, *d_site_min, *d_site_max;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S, cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_site_mean, out_S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_site_min, out_S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_site_max, out_S * sizeof(float)));

    gamma_smc_forward_site_summary_gpu(
        d_packed, n_words, d_pos, S,
        (float)mu_scalar, (float)rho_scalar, (float)Ne,
        d_pi, d_pj, n_pairs,
        d_site_mean, d_site_min, d_site_max, stride);

    auto mean_out = py::array_t<float>((ssize_t)out_S);
    auto min_out = py::array_t<float>((ssize_t)out_S);
    auto max_out = py::array_t<float>((ssize_t)out_S);
    CUDA_CHECK(cudaMemcpy(mean_out.mutable_data(), d_site_mean, out_S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(min_out.mutable_data(), d_site_min, out_S * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(max_out.mutable_data(), d_site_max, out_S * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_site_mean); cudaFree(d_site_min); cudaFree(d_site_max);

    py::dict result;
    result["site_mean"] = mean_out;
    result["site_min"] = min_out;
    result["site_max"] = max_out;
    result["n_pairs"] = n_pairs;
    result["out_S"] = out_S;
    return result;
}

// Gamma-SMC flow field forward-backward
// ============================================================

// Global: cached flow field data (loaded once)
static FlowFieldData g_flow_field;
static bool g_flow_field_loaded = false;
static float* g_d_flow_u = nullptr;
static float* g_d_flow_v = nullptr;

// Global: multi-step cache (rebuilt when params change)
static float* g_d_cache_mean = nullptr;
static float* g_d_cache_cv = nullptr;
static float2* g_d_cache_f2 = nullptr;   // interleaved (mean, cv) for fwd-only kernel
static void* g_d_cache_h2 = nullptr;    // half2 cache for fp16 kernel
static cudaArray_t g_cache_array = nullptr;   // layered CUDA array for texture
static cudaTextureObject_t g_cache_tex = 0;   // hardware bilinear texture
static int g_cache_tex_layers = 0;            // number of layers in texture
static int g_cache_n_steps = 0;
static float g_cache_rho = 0, g_cache_mu = 0, g_cache_Ne = 0;

// Forward-declare kernel launchers (defined in gamma_smc_flow.cu)
extern void gamma_smc_flow_tex_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    cudaTextureObject_t cache_tex, int n_tex_layers,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out);

extern void gamma_smc_flow_sync_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache, int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out);

extern void gamma_smc_flow_h2_fwd_gpu(
    const uint64_t* packed, int n_words,
    const double* positions, int S,
    float Ne,
    const int* pair_i, const int* pair_j, int n_pairs,
    const void* d_cache_h2, int n_max_steps,
    float* tmrca_mean_out,
    float* tmrca_lower_out,
    float* tmrca_upper_out);

static void ensure_flow_field(const std::string& path) {
    if (g_flow_field_loaded) return;
    if (!load_flow_field(path.c_str(), g_flow_field)) {
        throw std::runtime_error("Failed to load flow field from: " + path);
    }
    CUDA_CHECK(cudaMalloc(&g_d_flow_u, FF_GRID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_flow_v, FF_GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_flow_u, g_flow_field.u,
                          FF_GRID * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_flow_v, g_flow_field.v,
                          FF_GRID * sizeof(float), cudaMemcpyHostToDevice));
    g_flow_field_loaded = true;
}

static void ensure_cache(float Ne, float mu, float rho, int n_steps,
                          const std::string& ff_path) {
    ensure_flow_field(ff_path);

    // Rebuild if params changed
    if (g_d_cache_mean && g_cache_n_steps >= n_steps &&
        g_cache_rho == rho && g_cache_mu == mu && g_cache_Ne == Ne)
        return;

    // Free old
    if (g_d_cache_mean) { cudaFree(g_d_cache_mean); g_d_cache_mean = nullptr; }
    if (g_d_cache_cv)   { cudaFree(g_d_cache_cv); g_d_cache_cv = nullptr; }
    if (g_d_cache_f2)   { cudaFree(g_d_cache_f2); g_d_cache_f2 = nullptr; }
    if (g_d_cache_h2)   { cudaFree(g_d_cache_h2); g_d_cache_h2 = nullptr; }
    if (g_cache_tex)    { cudaDestroyTextureObject(g_cache_tex); g_cache_tex = 0; }
    if (g_cache_array)  { cudaFreeArray(g_cache_array); g_cache_array = nullptr; }
    g_cache_tex_layers = 0;

    float scaled_rho = 2.0f * Ne * rho;
    float scaled_mu  = 4.0f * Ne * mu;

    FlowFieldCache cache = build_flow_field_cache(g_flow_field, n_steps,
                                                   scaled_rho, scaled_mu);

    size_t total = (size_t)n_steps * FF_GRID;
    CUDA_CHECK(cudaMalloc(&g_d_cache_mean, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_cache_cv,   total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_cache_mean, cache.mean,
                          total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_cache_cv, cache.cv,
                          total * sizeof(float), cudaMemcpyHostToDevice));

    // Build interleaved float2 cache for fwd-only kernel
    std::vector<float> interleaved(total * 2);
    for (size_t i = 0; i < total; i++) {
        interleaved[2 * i]     = cache.mean[i];
        interleaved[2 * i + 1] = cache.cv[i];
    }
    CUDA_CHECK(cudaMalloc(&g_d_cache_f2, total * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(g_d_cache_f2, interleaved.data(),
                          total * sizeof(float2), cudaMemcpyHostToDevice));

    // Build half-precision (fp16) cache — halves L2 traffic per bilinear lookup
    {
        std::vector<__half2> h2_data(total);
        for (size_t i = 0; i < total; i++) {
            h2_data[i] = __floats2half2_rn(cache.mean[i], cache.cv[i]);
        }
        CUDA_CHECK(cudaMalloc(&g_d_cache_h2, total * sizeof(__half2)));
        CUDA_CHECK(cudaMemcpy(g_d_cache_h2, h2_data.data(),
                              total * sizeof(__half2), cudaMemcpyHostToDevice));
    }

    // Build layered 2D texture for hardware bilinear interpolation.
    // Max 2048 layers for layered 2D array; handle overflow via decomposition loop.
    {
        int n_layers = std::min(n_steps, 2048);
        g_cache_tex_layers = n_layers;

        // Reinterpret interleaved floats as float2 for host data
        const float2* f2_host = reinterpret_cast<const float2*>(interleaved.data());

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
        // width = FF_CV_N (cv dim), height = FF_MEAN_N (mean dim)
        cudaExtent extent = make_cudaExtent(FF_CV_N, FF_MEAN_N, n_layers);
        CUDA_CHECK(cudaMalloc3DArray(&g_cache_array, &desc, extent, cudaArrayLayered));

        // Copy host data → layered array
        cudaMemcpy3DParms p = {0};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)f2_host,
            FF_CV_N * sizeof(float2),   // pitch (row stride in bytes)
            FF_CV_N,                    // width in elements
            FF_MEAN_N                   // height in rows
        );
        p.dstArray = g_cache_array;
        p.extent = make_cudaExtent(FF_CV_N, FF_MEAN_N, n_layers);
        p.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&p));

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = g_cache_array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;  // cv dim
        texDesc.addressMode[1] = cudaAddressModeClamp;  // mean dim
        texDesc.filterMode = cudaFilterModeLinear;       // hardware bilinear!
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;                    // use texel coordinates

        CUDA_CHECK(cudaCreateTextureObject(&g_cache_tex, &resDesc, &texDesc, nullptr));
    }

    g_cache_n_steps = n_steps;
    g_cache_rho = rho;
    g_cache_mu = mu;
    g_cache_Ne = Ne;

    free_flow_field_cache(cache);
}

py::dict py_gamma_smc_flow_fb(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    std::string flow_field_path,
    bool mean_only)
{
    ensure_flow_field(flow_field_path);

    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double* d_pos;
    int *d_pi, *d_pj;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Determine chunk size: forward buffer = 2*S*chunk floats, output = 1-3*S*chunk floats
    int n_arrays = mean_only ? 1 : 3;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t per_pair = (size_t)S * (2 + n_arrays) * sizeof(float);  // fwd_buf + output
    size_t usable = (size_t)(free_mem * 0.7);
    int chunk_pairs = n_pairs;
    if (per_pair > 0 && (size_t)n_pairs * per_pair > usable) {
        chunk_pairs = std::max(1, (int)(usable / per_pair));
    }

    // Allocate GPU buffers for one chunk
    size_t chunk_sites = (size_t)chunk_pairs * S;
    float *d_fwd_buf, *d_mean, *d_lower = nullptr, *d_upper = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fwd_buf, 2 * chunk_sites * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, chunk_sites * sizeof(float)));
    if (!mean_only) {
        CUDA_CHECK(cudaMalloc(&d_lower, chunk_sites * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upper, chunk_sites * sizeof(float)));
    }

    // Host output: [S × n_pairs] site-major
    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    py::array_t<float> lower_out, upper_out;
    if (!mean_only) {
        lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
        upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    }

    float* h_mean = mean_out.mutable_data();
    float* h_lower = mean_only ? nullptr : lower_out.mutable_data();
    float* h_upper = mean_only ? nullptr : upper_out.mutable_data();

    // Process in chunks
    for (int offset = 0; offset < n_pairs; offset += chunk_pairs) {
        int chunk = std::min(chunk_pairs, n_pairs - offset);

        gamma_smc_flow_fb_gpu(
            d_packed, n_words, d_pos, S,
            (float)mu_scalar, (float)rho_scalar, (float)Ne,
            d_pi + offset, d_pj + offset, chunk,
            g_d_flow_u, g_d_flow_v,
            d_fwd_buf,
            d_mean, d_lower, d_upper);

        // Copy results to host
        if (chunk == n_pairs) {
            // No interleaving needed
            size_t bytes = (size_t)S * chunk * sizeof(float);
            CUDA_CHECK(cudaMemcpy(h_mean, d_mean, bytes, cudaMemcpyDeviceToHost));
            if (!mean_only) {
                CUDA_CHECK(cudaMemcpy(h_lower, d_lower, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_upper, d_upper, bytes, cudaMemcpyDeviceToHost));
            }
        } else {
            auto copy2d = [&](float* d_src, float* h_dst) {
                CUDA_CHECK(cudaMemcpy2D(
                    h_dst + offset,
                    (size_t)n_pairs * sizeof(float),
                    d_src,
                    (size_t)chunk * sizeof(float),
                    (size_t)chunk * sizeof(float),
                    S,
                    cudaMemcpyDeviceToHost));
            };
            copy2d(d_mean, h_mean);
            if (!mean_only) {
                copy2d(d_lower, h_lower);
                copy2d(d_upper, h_upper);
            }
        }
    }

    // Cleanup
    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_fwd_buf); cudaFree(d_mean);
    if (d_lower) cudaFree(d_lower);
    if (d_upper) cudaFree(d_upper);

    py::dict result;
    result["mean"] = mean_out;
    if (!mean_only) {
        result["lower"] = lower_out;
        result["upper"] = upper_out;
    }
    return result;
}

// ============================================================
// Gamma-SMC flow field forward-backward — CACHED (fast path)
// ============================================================
py::dict py_gamma_smc_flow_cached_fb(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    std::string flow_field_path,
    bool mean_only,
    int cache_steps)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    // Determine cache size from data if not specified
    if (cache_steps <= 0) {
        // Find max gap between consecutive sites
        const double* pos = (const double*)pos_buf.ptr;
        double max_gap = 0;
        for (int i = 1; i < S; i++) {
            double gap = pos[i] - pos[i - 1];
            if (gap > max_gap) max_gap = gap;
        }
        // Add 10% margin, minimum 1024
        cache_steps = std::max(1024, (int)(max_gap * 1.1) + 1);
        // Cap at 16384 to limit memory (16384 * 2550 * 4 * 2 = 314 MB)
        cache_steps = std::min(cache_steps, 16384);
    }

    // Build/upload cache (reuses if params match)
    ensure_cache((float)Ne, (float)mu_scalar, (float)rho_scalar, cache_steps,
                 flow_field_path);

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations
    uint8_t* d_G;
    uint64_t* d_packed;
    double* d_pos;
    int *d_pi, *d_pj;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Chunk sizing
    int n_arrays = mean_only ? 1 : 3;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t per_pair = (size_t)S * (2 + n_arrays) * sizeof(float);
    size_t usable = (size_t)(free_mem * 0.7);
    int chunk_pairs = n_pairs;
    if (per_pair > 0 && (size_t)n_pairs * per_pair > usable)
        chunk_pairs = std::max(1, (int)(usable / per_pair));

    size_t chunk_sites = (size_t)chunk_pairs * S;
    float *d_fwd_buf, *d_mean, *d_lower = nullptr, *d_upper = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fwd_buf, 2 * chunk_sites * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, chunk_sites * sizeof(float)));
    if (!mean_only) {
        CUDA_CHECK(cudaMalloc(&d_lower, chunk_sites * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upper, chunk_sites * sizeof(float)));
    }

    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    py::array_t<float> lower_out, upper_out;
    if (!mean_only) {
        lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
        upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    }

    float* h_mean = mean_out.mutable_data();
    float* h_lower = mean_only ? nullptr : lower_out.mutable_data();
    float* h_upper = mean_only ? nullptr : upper_out.mutable_data();

    for (int offset = 0; offset < n_pairs; offset += chunk_pairs) {
        int chunk = std::min(chunk_pairs, n_pairs - offset);

        gamma_smc_flow_cached_fb_gpu(
            d_packed, n_words, d_pos, S, (float)Ne,
            d_pi + offset, d_pj + offset, chunk,
            g_d_cache_mean, g_d_cache_cv, g_cache_n_steps,
            d_fwd_buf,
            d_mean, d_lower, d_upper);

        if (chunk == n_pairs) {
            size_t bytes = (size_t)S * chunk * sizeof(float);
            CUDA_CHECK(cudaMemcpy(h_mean, d_mean, bytes, cudaMemcpyDeviceToHost));
            if (!mean_only) {
                CUDA_CHECK(cudaMemcpy(h_lower, d_lower, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_upper, d_upper, bytes, cudaMemcpyDeviceToHost));
            }
        } else {
            auto copy2d = [&](float* d_src, float* h_dst) {
                CUDA_CHECK(cudaMemcpy2D(
                    h_dst + offset, (size_t)n_pairs * sizeof(float),
                    d_src, (size_t)chunk * sizeof(float),
                    (size_t)chunk * sizeof(float), S,
                    cudaMemcpyDeviceToHost));
            };
            copy2d(d_mean, h_mean);
            if (!mean_only) {
                copy2d(d_lower, h_lower);
                copy2d(d_upper, h_upper);
            }
        }
    }

    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_fwd_buf); cudaFree(d_mean);
    if (d_lower) cudaFree(d_lower);
    if (d_upper) cudaFree(d_upper);

    py::dict result;
    result["mean"] = mean_out;
    if (!mean_only) {
        result["lower"] = lower_out;
        result["upper"] = upper_out;
    }
    return result;
}

// ============================================================
// FlowContext: persistent GPU context for max throughput
// ============================================================
class FlowContext {
    uint64_t* d_packed_ = nullptr;
    double* d_pos_ = nullptr;
    int* d_pi_ = nullptr;
    int* d_pj_ = nullptr;
    float* d_mean_ = nullptr;
    float* d_lower_ = nullptr;
    float* d_upper_ = nullptr;

    int n_haps_, n_words_, S_;
    int max_pairs_;      // current device output allocation size
    bool has_ci_;
    float* d_fwd_buf_ = nullptr;
    int fwd_buf_pairs_ = 0;  // how many pairs the fwd_buf can hold
    int device_id_ = 0;  // GPU device this context lives on
    float* ctx_cache_mean_ = nullptr;
    float* ctx_cache_cv_ = nullptr;
    void* ctx_cache_h2_ = nullptr;
    int ctx_cache_steps_ = 0;
    float ctx_cache_Ne_ = 0;

    void alloc_output(int n_pairs, bool ci) {
        if (n_pairs <= max_pairs_ && ci == has_ci_) return;
        free_output();
        max_pairs_ = n_pairs;
        has_ci_ = ci;

        size_t total = (size_t)n_pairs * S_;
        CUDA_CHECK(cudaMalloc(&d_mean_, total * sizeof(float)));
        if (ci) {
            CUDA_CHECK(cudaMalloc(&d_lower_, total * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_upper_, total * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_pi_, n_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pj_, n_pairs * sizeof(int)));
    }

    void alloc_fwd_buf(int n_pairs) {
        if (n_pairs <= fwd_buf_pairs_) return;
        if (d_fwd_buf_) { cudaFree(d_fwd_buf_); d_fwd_buf_ = nullptr; }
        size_t bytes = 2ULL * S_ * n_pairs * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_fwd_buf_, bytes));
        fwd_buf_pairs_ = n_pairs;
    }

    void free_output() {
        if (d_mean_) { cudaFree(d_mean_); d_mean_ = nullptr; }
        if (d_lower_) { cudaFree(d_lower_); d_lower_ = nullptr; }
        if (d_upper_) { cudaFree(d_upper_); d_upper_ = nullptr; }
        if (d_pi_) { cudaFree(d_pi_); d_pi_ = nullptr; }
        if (d_pj_) { cudaFree(d_pj_); d_pj_ = nullptr; }
        max_pairs_ = 0;
    }

public:
    FlowContext(
        py::array_t<uint8_t, py::array::c_style> G,
        py::array_t<double, py::array::c_style> positions_arr,
        double Ne, double mu, double rho,
        std::string flow_field_path,
        int cache_steps)
    {
        auto g_buf = G.request();
        auto pos_buf = positions_arr.request();

        cudaGetDevice(&device_id_);  // remember which GPU we're on
        n_haps_ = (int)g_buf.shape[0];
        S_ = (int)g_buf.shape[1];
        n_words_ = (S_ + 63) / 64;
        max_pairs_ = 0;
        has_ci_ = false;

        // Determine cache steps from positions
        if (cache_steps <= 0) {
            const double* pos = (const double*)pos_buf.ptr;
            double max_gap = 0;
            for (int i = 1; i < S_; i++) {
                double gap = pos[i] - pos[i - 1];
                if (gap > max_gap) max_gap = gap;
            }
            cache_steps = std::max(1024, (int)(max_gap * 1.1) + 1);
            cache_steps = std::min(cache_steps, 16384);
        }

        // Build/upload cache
        ensure_cache((float)Ne, (float)mu, (float)rho, cache_steps, flow_field_path);

        // Multi-GPU: each device needs its own cache allocation.
        // Save global pointers, force rebuild on this device, then restore.
        {
            float* saved_mean = g_d_cache_mean;
            float* saved_cv = g_d_cache_cv;
            float2* saved_f2 = g_d_cache_f2;
            void* saved_h2 = g_d_cache_h2;
            int saved_steps = g_cache_n_steps;
            
            // Check if cache is on a different device
            int cache_device = -1;
            cudaPointerAttributes attr;
            if (saved_mean && cudaPointerGetAttributes(&attr, saved_mean) == cudaSuccess) {
                cache_device = attr.device;
            }
            cudaGetLastError();
            
            if (cache_device != device_id_ && cache_device >= 0) {
                // Force rebuild: temporarily null the globals so ensure_cache rebuilds
                g_d_cache_mean = nullptr;
                g_d_cache_cv = nullptr;
                g_d_cache_f2 = nullptr;
                g_d_cache_h2 = nullptr;
                g_cache_n_steps = 0;
                
                ensure_cache((float)Ne, (float)mu, (float)rho, cache_steps, flow_field_path);
                
                // Save the device-local pointers
                ctx_cache_mean_ = g_d_cache_mean;
                ctx_cache_cv_ = g_d_cache_cv;
                ctx_cache_h2_ = g_d_cache_h2;
                ctx_cache_steps_ = g_cache_n_steps;
                ctx_cache_Ne_ = g_cache_Ne;
                
                // Restore globals (so the original device's cache isn't lost)
                g_d_cache_mean = saved_mean;
                g_d_cache_cv = saved_cv;
                g_d_cache_f2 = saved_f2;
                g_d_cache_h2 = saved_h2;
                g_cache_n_steps = saved_steps;
            } else {
                // Same device or first time: use globals directly
                ctx_cache_mean_ = g_d_cache_mean;
                ctx_cache_cv_ = g_d_cache_cv;
                ctx_cache_h2_ = g_d_cache_h2;
                ctx_cache_steps_ = g_cache_n_steps;
                ctx_cache_Ne_ = g_cache_Ne;
            }
        }
        // Upload and bitpack genotypes
        uint8_t* d_G;
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)n_haps_ * S_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_packed_, (size_t)n_haps_ * n_words_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_packed_, 0, (size_t)n_haps_ * n_words_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n_haps_ * S_, cudaMemcpyHostToDevice));
        bitpack_genotypes_gpu(d_G, d_packed_, n_haps_, S_, n_words_);
        cudaFree(d_G);

        // Upload positions
        CUDA_CHECK(cudaMalloc(&d_pos_, S_ * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_pos_, pos_buf.ptr, S_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~FlowContext() {
        free_output();
        if (d_fwd_buf_) cudaFree(d_fwd_buf_);
        if (d_packed_) cudaFree(d_packed_);
        if (d_pos_) cudaFree(d_pos_);
    }

    FlowContext(const FlowContext&) = delete;
    FlowContext& operator=(const FlowContext&) = delete;

    py::dict run_fb_summary(std::vector<std::pair<int, int>> pairs) {
        int n_pairs = (int)pairs.size();
        if (n_pairs == 0) {
            py::dict result;
            result["site_mean"] = py::array_t<float>(0);
            return result;
        }

        std::vector<int> pi(n_pairs), pj(n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            pi[p] = pairs[p].first;
            pj[p] = pairs[p].second;
        }

        auto mean_out = py::array_t<float>((ssize_t)S_);
        auto min_out = py::array_t<float>((ssize_t)S_);
        auto max_out = py::array_t<float>((ssize_t)S_);
        float* h_mean = mean_out.mutable_data();
        float* h_min = min_out.mutable_data();
        float* h_max = max_out.mutable_data();

        {
            py::gil_scoped_release release;
            cudaSetDevice(device_id_);
            alloc_fwd_buf(n_pairs);
            alloc_output(n_pairs, false);

            CUDA_CHECK(cudaMemcpy(d_pi_, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pj_, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

            float *d_site_mean, *d_site_min, *d_site_max;
            CUDA_CHECK(cudaMalloc(&d_site_mean, S_ * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_site_min, S_ * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_site_max, S_ * sizeof(float)));

            gamma_smc_flow_cached_fb_reduce_gpu(
                d_packed_, n_words_, d_pos_, S_, ctx_cache_Ne_,
                d_pi_, d_pj_, n_pairs,
                ctx_cache_mean_, ctx_cache_cv_, ctx_cache_steps_,
                d_fwd_buf_,
                d_site_mean, d_site_min, d_site_max);

            CUDA_CHECK(cudaMemcpy(h_mean, d_site_mean, S_ * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min, d_site_min, S_ * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_max, d_site_max, S_ * sizeof(float), cudaMemcpyDeviceToHost));

            cudaFree(d_site_mean);
            cudaFree(d_site_min);
            cudaFree(d_site_max);
        }

        py::dict result;
        result["site_mean"] = mean_out;
        result["site_min"] = min_out;
        result["site_max"] = max_out;
        result["n_pairs"] = n_pairs;
        return result;
    }

    int n_haps() const { return n_haps_; }
    int n_sites() const { return S_; }
    int device_id() const { return device_id_; }

    py::dict run_fwd(std::vector<std::pair<int, int>> pairs, bool mean_only) {
        int n_pairs = (int)pairs.size();
        if (n_pairs == 0) {
            py::dict result;
            result["mean"] = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, 0});
            return result;
        }

        bool ci = !mean_only;
        size_t bytes = (size_t)S_ * n_pairs * sizeof(float);

        std::vector<int> pi(n_pairs), pj(n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            pi[p] = pairs[p].first;
            pj[p] = pairs[p].second;
        }

        auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
        py::array_t<float> lower_out, upper_out;
        if (ci) {
            lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
            upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
        }
        float* h_mean = mean_out.mutable_data();
        float* h_lower = ci ? lower_out.mutable_data() : nullptr;
        float* h_upper = ci ? upper_out.mutable_data() : nullptr;

        {
            py::gil_scoped_release release;
            cudaSetDevice(device_id_);
            alloc_output(n_pairs, ci);

            CUDA_CHECK(cudaMemcpy(d_pi_, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pj_, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

            gamma_smc_flow_h2_fwd_gpu(
                d_packed_, n_words_, d_pos_, S_, ctx_cache_Ne_,
                d_pi_, d_pj_, n_pairs,
                ctx_cache_h2_, ctx_cache_steps_,
                d_mean_, ci ? d_lower_ : nullptr, ci ? d_upper_ : nullptr);

            CUDA_CHECK(cudaMemcpy(h_mean, d_mean_, bytes, cudaMemcpyDeviceToHost));
            if (ci) {
                CUDA_CHECK(cudaMemcpy(h_lower, d_lower_, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_upper, d_upper_, bytes, cudaMemcpyDeviceToHost));
            }
        }

        py::dict result;
        result["mean"] = mean_out;
        if (ci) {
            result["lower"] = lower_out;
            result["upper"] = upper_out;
        }
        return result;
    }

    py::dict run_fb(std::vector<std::pair<int, int>> pairs, bool mean_only) {
        int n_pairs = (int)pairs.size();
        if (n_pairs == 0) {
            py::dict result;
            result["mean"] = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, 0});
            return result;
        }

        bool ci = !mean_only;
        size_t bytes = (size_t)S_ * n_pairs * sizeof(float);

        std::vector<int> pi(n_pairs), pj(n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            pi[p] = pairs[p].first;
            pj[p] = pairs[p].second;
        }

        auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
        py::array_t<float> lower_out, upper_out;
        if (ci) {
            lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
            upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S_, (ssize_t)n_pairs});
        }
        float* h_mean = mean_out.mutable_data();
        float* h_lower = ci ? lower_out.mutable_data() : nullptr;
        float* h_upper = ci ? upper_out.mutable_data() : nullptr;

        {
            py::gil_scoped_release release;
            cudaSetDevice(device_id_);

            size_t fwd_buf_bytes = 2ULL * S_ * n_pairs * sizeof(float);
            float* d_fwd_buf;
            CUDA_CHECK(cudaMalloc(&d_fwd_buf, fwd_buf_bytes));
            alloc_output(n_pairs, ci);

            CUDA_CHECK(cudaMemcpy(d_pi_, pi.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pj_, pj.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice));

            gamma_smc_flow_cached_fb_gpu(
                d_packed_, n_words_, d_pos_, S_, ctx_cache_Ne_,
                d_pi_, d_pj_, n_pairs,
                ctx_cache_mean_, ctx_cache_cv_, ctx_cache_steps_,
                d_fwd_buf,
                d_mean_, ci ? d_lower_ : nullptr, ci ? d_upper_ : nullptr);

            cudaFree(d_fwd_buf);

            CUDA_CHECK(cudaMemcpy(h_mean, d_mean_, bytes, cudaMemcpyDeviceToHost));
            if (ci) {
                CUDA_CHECK(cudaMemcpy(h_lower, d_lower_, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_upper, d_upper_, bytes, cudaMemcpyDeviceToHost));
            }
        }

        py::dict result;
        result["mean"] = mean_out;
        if (ci) {
            result["lower"] = lower_out;
            result["upper"] = upper_out;
        }
        return result;
    }
};

// ============================================================
// Forward-only with cache — no forward buffer, max throughput
// ============================================================
py::dict py_gamma_smc_flow_cached_fwd(
    py::array_t<uint8_t, py::array::c_style> G,
    py::array_t<double, py::array::c_style> positions_arr,
    std::vector<std::pair<int, int>> pairs,
    double Ne,
    double mu_scalar,
    double rho_scalar,
    std::string flow_field_path,
    bool mean_only,
    int cache_steps)
{
    auto g_buf = G.request();
    auto pos_buf = positions_arr.request();

    int n = (int)g_buf.shape[0];
    int S = (int)g_buf.shape[1];
    int n_words = (S + 63) / 64;
    int n_pairs = (int)pairs.size();

    // Determine cache size from data if not specified
    if (cache_steps <= 0) {
        const double* pos = (const double*)pos_buf.ptr;
        double max_gap = 0;
        for (int i = 1; i < S; i++) {
            double gap = pos[i] - pos[i - 1];
            if (gap > max_gap) max_gap = gap;
        }
        cache_steps = std::max(1024, (int)(max_gap * 1.1) + 1);
        cache_steps = std::min(cache_steps, 16384);
    }

    ensure_cache((float)Ne, (float)mu_scalar, (float)rho_scalar, cache_steps,
                 flow_field_path);

    // Pair indices
    std::vector<int> pi(n_pairs), pj(n_pairs);
    for (int p = 0; p < n_pairs; p++) {
        pi[p] = pairs[p].first;
        pj[p] = pairs[p].second;
    }

    // GPU allocations: genotypes + positions + pairs
    uint8_t* d_G;
    uint64_t* d_packed;
    double* d_pos;
    int *d_pi, *d_pj;

    CUDA_CHECK(cudaMalloc(&d_G, (size_t)n * S * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_packed, 0, (size_t)n * n_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_G, g_buf.ptr, (size_t)n * S * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    bitpack_genotypes_gpu(d_G, d_packed, n, S, n_words);
    cudaFree(d_G);

    CUDA_CHECK(cudaMalloc(&d_pos, S * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_buf.ptr, S * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pi, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj, n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pi, pi.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj, pj.data(), n_pairs * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Output-only allocation (NO forward buffer!)
    int n_arrays = mean_only ? 1 : 3;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t per_pair = (size_t)S * n_arrays * sizeof(float);
    size_t usable = (size_t)(free_mem * 0.7);
    int chunk_pairs = n_pairs;
    if (per_pair > 0 && (size_t)n_pairs * per_pair > usable)
        chunk_pairs = std::max(1, (int)(usable / per_pair));

    size_t chunk_sites = (size_t)chunk_pairs * S;
    float *d_mean, *d_lower = nullptr, *d_upper = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mean, chunk_sites * sizeof(float)));
    if (!mean_only) {
        CUDA_CHECK(cudaMalloc(&d_lower, chunk_sites * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upper, chunk_sites * sizeof(float)));
    }

    auto mean_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    py::array_t<float> lower_out, upper_out;
    if (!mean_only) {
        lower_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
        upper_out = py::array_t<float>(std::vector<ssize_t>{(ssize_t)S, (ssize_t)n_pairs});
    }

    float* h_mean = mean_out.mutable_data();
    float* h_lower = mean_only ? nullptr : lower_out.mutable_data();
    float* h_upper = mean_only ? nullptr : upper_out.mutable_data();

    for (int offset = 0; offset < n_pairs; offset += chunk_pairs) {
        int chunk = std::min(chunk_pairs, n_pairs - offset);

        if (g_cache_tex) {
            gamma_smc_flow_tex_fwd_gpu(
                d_packed, n_words, d_pos, S, (float)Ne,
                d_pi + offset, d_pj + offset, chunk,
                g_cache_tex, g_cache_tex_layers,
                d_mean, d_lower, d_upper);
        } else {
            gamma_smc_flow_cached_fwd_gpu(
                d_packed, n_words, d_pos, S, (float)Ne,
                d_pi + offset, d_pj + offset, chunk,
                g_d_cache_f2, g_cache_n_steps,
                d_mean, d_lower, d_upper);
        }

        if (chunk == n_pairs) {
            size_t bytes = (size_t)S * chunk * sizeof(float);
            CUDA_CHECK(cudaMemcpy(h_mean, d_mean, bytes, cudaMemcpyDeviceToHost));
            if (!mean_only) {
                CUDA_CHECK(cudaMemcpy(h_lower, d_lower, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_upper, d_upper, bytes, cudaMemcpyDeviceToHost));
            }
        } else {
            auto copy2d = [&](float* d_src, float* h_dst) {
                CUDA_CHECK(cudaMemcpy2D(
                    h_dst + offset, (size_t)n_pairs * sizeof(float),
                    d_src, (size_t)chunk * sizeof(float),
                    (size_t)chunk * sizeof(float), S,
                    cudaMemcpyDeviceToHost));
            };
            copy2d(d_mean, h_mean);
            if (!mean_only) {
                copy2d(d_lower, h_lower);
                copy2d(d_upper, h_upper);
            }
        }
    }

    cudaFree(d_packed); cudaFree(d_pos);
    cudaFree(d_pi); cudaFree(d_pj);
    cudaFree(d_mean);
    if (d_lower) cudaFree(d_lower);
    if (d_upper) cudaFree(d_upper);

    py::dict result;
    result["mean"] = mean_out;
    if (!mean_only) {
        result["lower"] = lower_out;
        result["upper"] = upper_out;
    }
    return result;
}

// ============================================================
// Module definition
// ============================================================
PYBIND11_MODULE(_core, m) {
    m.doc() = "tmrca_cu: GPU-accelerated pairwise coalescence time estimation";

    m.def("bitpack", &py_bitpack,
          "Bitpack a genotype matrix G[n,S] into uint64 words",
          py::arg("G"));

    m.def("unpack", &py_unpack,
          "Unpack bitpacked matrix back to genotype matrix",
          py::arg("packed"), py::arg("n"), py::arg("S"));

    m.def("pairwise_prefix_scan", &py_pairwise_prefix_scan,
          "Compute pairwise XOR prefix scan (cumulative difference count)",
          py::arg("G"), py::arg("pairs"));

    m.def("windowed_divergence", &py_windowed_divergence,
          "Compute windowed pairwise divergence",
          py::arg("G"), py::arg("pairs"), py::arg("window_sites"));

    m.def("compute_sfs", &py_compute_sfs,
          "Compute site frequency spectrum from genotype matrix",
          py::arg("G"));

    m.def("coalescent_prior", &py_coalescent_prior,
          "Compute coalescent prior q[k] under constant Ne",
          py::arg("Ne"), py::arg("K") = 32, py::arg("t_max") = -1.0);

    m.def("hmm_posterior", &py_hmm_posterior,
          "Run HMM forward-backward and return posterior marginals (S x K)",
          py::arg("G"), py::arg("positions"), py::arg("pair"),
          py::arg("K") = 32, py::arg("Ne") = 10000.0,
          py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
          py::arg("t_max") = -1.0);

    m.def("hmm_log_likelihood", &py_hmm_log_likelihood,
          "Run HMM forward-backward and return total log-likelihood",
          py::arg("G"), py::arg("positions"), py::arg("pair"),
          py::arg("K") = 32, py::arg("Ne") = 10000.0,
          py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
          py::arg("t_max") = -1.0);

    m.def("hmm_posterior_batched", &py_hmm_posterior_batched,
          "Run batched HMM for multiple pairs, return (gamma, mean, lower, upper, loglik)",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("K") = 32, py::arg("Ne") = 10000.0,
          py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
          py::arg("t_max") = -1.0);

    m.def("time_midpoints", &py_time_midpoints,
          "Get time bin midpoints for given K, Ne, t_max",
          py::arg("K") = 32, py::arg("Ne") = 10000.0, py::arg("t_max") = -1.0);

    m.def("time_boundaries", &py_time_boundaries,
          "Get time bin boundaries for given K, Ne, t_max",
          py::arg("K") = 32, py::arg("Ne") = 10000.0, py::arg("t_max") = -1.0);

    m.def("site_pi", &py_site_pi,
          "Compute per-site nucleotide diversity pi(s) using GPU",
          py::arg("G"), py::arg("n_sample_pairs") = 1000);

    m.def("pelt_changepoint", &py_pelt_changepoint,
          "Run PELT changepoint detection on GPU.\n"
          "Takes prefix scan array (n_pairs x S), positions (S,), n_pairs, mu, penalty.\n"
          "Returns dict with n_segments, seg_starts, seg_ends, seg_tmrca, seg_counts.",
          py::arg("prefix"), py::arg("positions"),
          py::arg("n_pairs"), py::arg("mu"), py::arg("penalty"));

    m.def("ep_infer", &py_ep_infer,
          "Run EP inference: HMM + ultrametric loop on GPU.\n"
          "Returns dict with gamma, mean, lower, upper, log_likelihood, converged, n_iterations, ll_history.",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("m_haplotypes"),
          py::arg("K") = 32, py::arg("Ne") = 10000.0,
          py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
          py::arg("t_max") = -1.0,
          py::arg("max_iterations") = 5,
          py::arg("damping") = 0.5,
          py::arg("convergence_tol") = 0.01);

    m.def("adaptive_prior_infer", &py_adaptive_prior_infer,
          "Run adaptive prior inference: EM loop (HMM → aggregate → update prior → repeat).\n"
          "Returns dict with gamma, mean, lower, upper, log_likelihood, prior, converged, n_iterations, ll_history.",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("K") = 32, py::arg("Ne") = 10000.0,
          py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
          py::arg("t_max") = -1.0,
          py::arg("max_iterations") = 20,
          py::arg("blend_alpha") = 0.7,
          py::arg("convergence_tol") = 1e-6);

    m.def("gamma_smc_forward", &py_gamma_smc_forward,
          "Gamma-SMC forward filtering on GPU.\n"
          "Returns dict with 'mean' (and 'lower','upper' unless mean_only=True).\n"
          "Arrays are Fortran-order [n_pairs, out_S].",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
          py::arg("rho") = 1e-8, py::arg("stride") = 1,
          py::arg("mean_only") = false);

    m.def("gamma_smc_flow_fb", &py_gamma_smc_flow_fb,
          "Gamma-SMC forward-backward with flow field transitions on GPU.\n"
          "Uses Schweiger's precomputed flow field for exact recombination transitions.\n"
          "Returns dict with 'mean' (and 'lower','upper' unless mean_only=True).\n"
          "Arrays are site-major [S, n_pairs].",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
          py::arg("rho") = 1e-8,
          py::arg("flow_field_path") = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt",
          py::arg("mean_only") = false);

    m.def("gamma_smc_flow_cached_fwd", &py_gamma_smc_flow_cached_fwd,
          "Gamma-SMC forward-only with precomputed multi-step cache on GPU.\n"
          "No forward buffer — single pass, maximum throughput.\n"
          "Returns dict with 'mean' (and 'lower','upper' unless mean_only=True).",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
          py::arg("rho") = 1e-8,
          py::arg("flow_field_path") = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt",
          py::arg("mean_only") = true,
          py::arg("cache_steps") = 0);

    m.def("gamma_smc_flow_cached_fb", &py_gamma_smc_flow_cached_fb,
          "Gamma-SMC forward-backward with precomputed multi-step cache on GPU.\n"
          "Eliminates per-site flow field iteration — single bilinear lookup per site.\n"
          "Cache is built on first call and reused for same (Ne, mu, rho).\n"
          "Returns dict with 'mean' (and 'lower','upper' unless mean_only=True).",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
          py::arg("rho") = 1e-8,
          py::arg("flow_field_path") = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt",
          py::arg("mean_only") = false,
          py::arg("cache_steps") = 0);

    m.def("set_device", [](int device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }, py::arg("device_id"), "Set the active CUDA device");

    m.def("get_device_count", []() -> int {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        return count;
    }, "Get the number of available CUDA devices");

    m.def("get_device", []() -> int {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        return dev;
    }, "Get the current CUDA device");

    m.def("gamma_smc_site_summary", &py_gamma_smc_site_summary,
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne"), py::arg("mu"), py::arg("rho"),
          py::arg("stride") = 1);

    m.def("gamma_smc_forward_quantized", &py_gamma_smc_forward_quantized,
          "Gamma-SMC forward filtering with log-scale quantized output.\n"
          "Returns dict with 'quantized' (uint8 array), 'bits', 'log_min', 'log_max'.\n"
          "Dequantize: t = exp(log_min + (q / (2^bits-1)) * (log_max - log_min))",
          py::arg("G"), py::arg("positions"), py::arg("pairs"),
          py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
          py::arg("rho") = 1e-8, py::arg("stride") = 1,
          py::arg("bits") = 8);

    py::class_<HMMContext>(m, "HMMContext",
        "Persistent GPU context for batched HMM inference.\n"
        "Holds packed genotypes and parameter arrays on GPU to avoid\n"
        "re-uploading per batch call. Auto-chunks pairs to fit VRAM.")
        .def(py::init<py::array_t<uint8_t, py::array::c_style>,
                       py::array_t<double, py::array::c_style>,
                       int, double, double, double, double>(),
             py::arg("G"), py::arg("positions"),
             py::arg("K") = 32, py::arg("Ne") = 10000.0,
             py::arg("mu") = 1.25e-8, py::arg("rho") = 1e-8,
             py::arg("t_max") = -1.0)
        .def("run_batch", &HMMContext::run_batch,
             "Run HMM forward-backward for pairs. Returns (mean, lower, upper, loglik).\n"
             "Auto-chunks to fit GPU memory.",
             py::arg("pairs"))
        .def_property_readonly("n_haps", &HMMContext::n_haps)
        .def_property_readonly("n_sites", &HMMContext::n_sites)
        .def_property_readonly("n_bins", &HMMContext::n_bins);

    py::class_<FlowContext>(m, "FlowContext",
        "Persistent GPU context for flow-field Gamma-SMC inference.\n"
        "Holds packed genotypes, positions, and cache on GPU.\n"
        "Eliminates per-call allocation overhead for max throughput.\n"
        "Use run_fwd() for forward-only (fastest) or run_fb() for smoothed.")
        .def(py::init<py::array_t<uint8_t, py::array::c_style>,
                       py::array_t<double, py::array::c_style>,
                       double, double, double, std::string, int>(),
             py::arg("G"), py::arg("positions"),
             py::arg("Ne") = 10000.0, py::arg("mu") = 1.25e-8,
             py::arg("rho") = 1e-8,
             py::arg("flow_field_path") = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt",
             py::arg("cache_steps") = 0)
        .def("run_fwd", &FlowContext::run_fwd,
             "Forward-only filtering (fastest). Returns dict with 'mean'.\n"
             "No forward buffer needed — single pass, pinned D2H.",
             py::arg("pairs"), py::arg("mean_only") = true)
        .def("run_fb", &FlowContext::run_fb,
             "Forward-backward smoothing. Returns dict with 'mean'.",
             py::arg("pairs"), py::arg("mean_only") = true)
        .def("run_fb_summary", &FlowContext::run_fb_summary, py::arg("pairs"))
        .def_property_readonly("device_id", &FlowContext::device_id)
        .def_property_readonly("n_haps", &FlowContext::n_haps)
        .def_property_readonly("n_sites", &FlowContext::n_sites);
}
