tive Pairwise Coalescence Time Estimation at Biobank Scale

## Vision

A CUDA library that estimates pairwise coalescence times (TMRCA) at every genomic site, directly from a genotype matrix, without ever inferring a tree sequence or ARG. The method achieves MSMC2-tier accuracy through a novel variational inference scheme that decomposes the ARG posterior into two GPU-parallel factors: an along-genome SMC hidden Markov model (per pair) and an across-pair ultrametric tree projection (per site), coupled via expectation propagation.

The result is a **coalescence time field** $T: \text{pairs} \times \text{sites} \to \mathbb{R}_+$ — an object that no existing tool produces at scale.

## Naming Conventions

| Context | Name | Notes |
|---------|------|-------|
| Canonical name | `gamma_smc_cu` | Used in conversation, paper, docs |
| GitHub repository | `kevinkorfmann/gamma_smc_cu` | GitHub allows dots in repo names |
| PyPI package | `tmrca-cu` | `pip install tmrca-cu` (PyPI convention: hyphens) |
| Python import | `gamma_smc_cu` | `import gamma_smc_cu` (Python convention: underscores) |
| CMake project name | `gamma_smc_cu` | No punctuation in build system identifiers |
| C/CUDA library | `libgamma_smc_cu` | Shared library: `libgamma_smc_cu.so` |
| Conda package | `tmrca-cu` | Match PyPI name |

---

## 1. Mathematical Foundation

### 1.1 Problem Statement

**Input:**
- Genotype matrix $G \in \{0, 1\}^{n \times S}$ where $n$ = number of haploid samples, $S$ = number of segregating sites
- Physical positions $\mathbf{x} \in \mathbb{R}^S$ (base-pair coordinates of each segregating site)
- Per-site or regional mutation rate map $\mu(s)$ (can be uniform as fallback)
- Recombination rate map $\rho(s)$ (per-base-pair recombination rate between adjacent sites)

**Output:**
- For a target set of pairs $\mathcal{P} \subseteq \binom{[n]}{2}$, posterior mean and credible intervals of $T_{ij}(s)$ for all $(i,j) \in \mathcal{P}$ and all sites $s \in [S]$
- Per-site summary statistics: mean TMRCA, TMRCA distribution, $\pi(s)$

**Scale targets:**
- $n$ up to $10^6$ haplotypes (500K diploid individuals)
- $S$ up to $10^8$ sites (whole genome, all segregating sites)
- Wall-clock: minutes for Tier 1/2, hours for Tier 3 on a single multi-GPU node (8×A100/H100)

### 1.2 The Sequentially Markov Coalescent (SMC) for a Single Pair

For a pair of haplotypes $(i, j)$, the SMC models the TMRCA $T_{ij}(s)$ as a Markov chain along the genome. We discretize time into $K$ bins with boundaries $\{t_0 = 0, t_1, \ldots, t_K\}$, where bin $k$ represents the interval $[t_k, t_{k+1})$ with midpoint $\bar{t}_k = (t_k + t_{k+1}) / 2$.

**Time discretization (logarithmic):**

$$t_k = t_{\max} \cdot \left(\frac{k}{K}\right)^2, \quad k = 0, 1, \ldots, K$$

where $t_{\max}$ is the maximum coalescence time considered (e.g., $10 \cdot N_e$ generations). The quadratic spacing gives finer resolution for recent times where the data is most informative. We use $K = 32$ bins as the default — this fits the transition matrix in GPU registers and provides sufficient resolution.

**Emission probabilities:**

At each segregating site $s$, the observed data is the binary indicator $d_s = g_{i,s} \oplus g_{j,s}$ (XOR of the two haplotypes). Under the infinite-sites model:

$$P(d_s = 1 \mid T_{ij}(s) = \bar{t}_k) = 1 - e^{-2 \mu(s) \bar{t}_k}$$
$$P(d_s = 0 \mid T_{ij}(s) = \bar{t}_k) = e^{-2 \mu(s) \bar{t}_k}$$

But we must also account for sites between segregating sites — stretches of $L$ base pairs with no mutations in either haplotype. For a span of $L$ base pairs with average mutation rate $\bar{\mu}$ and no observed differences:

$$P(\text{no mutations in } L \text{ bp} \mid T = \bar{t}_k) = e^{-2 \bar{\mu} \bar{t}_k L}$$

Implementation note: In the HMM, between consecutive segregating sites at positions $x_s$ and $x_{s+1}$, we have a gap of $L = x_{s+1} - x_s - 1$ base pairs with no mutations. The emission at the transition between sites $s$ and $s+1$ must include this "gap emission" factor.

**Transition probabilities (SMC kernel):**

The probability of transitioning from TMRCA bin $k$ to bin $l$ between two loci separated by $r$ Morgans of recombination distance:

$$A_{kl}(r) = \begin{cases}
e^{-r \bar{t}_k} + (1 - e^{-r \bar{t}_k}) \cdot q_l & \text{if } k = l \\
(1 - e^{-r \bar{t}_k}) \cdot q_l & \text{if } k \neq l
\end{cases}$$

where:
- $e^{-r \bar{t}_k}$ is the probability of no recombination event in the ancestral lineage (the pair stays on the same genealogical tree)
- $q_l$ is the probability of coalescing in time bin $l$ under the demographic model (the coalescent prior), given that a recombination event occurred

The coalescent prior under piecewise-constant population size $N_e(t)$:

$$q_k = \int_{t_k}^{t_{k+1}} \frac{1}{N_e(t)} \exp\left(-\int_0^t \frac{1}{N_e(u)} du\right) dt$$

This is the probability of coalescence in time bin $k$ for two lineages starting at time 0. Under piecewise-constant $N_e$, this integral has a closed-form solution involving sums of exponentials.

**Recombination distance between sites:**

$$r_{s \to s+1} = \sum_{b=x_s}^{x_{s+1}-1} \rho(b) \approx \bar{\rho}_{s,s+1} \cdot (x_{s+1} - x_s)$$

where $\bar{\rho}_{s,s+1}$ is the average per-base recombination rate between sites $s$ and $s+1$.

### 1.3 Demographic Prior Estimation

Before running the HMM, we estimate the piecewise-constant population size history $N_e(t)$ from genome-wide summary statistics. This serves as the coalescent prior.

**Method:** Use the site frequency spectrum (SFS) computed from the genotype matrix. For $n$ haplotypes, the SFS entry $\xi_k$ (number of sites with derived allele count $k$) is related to $N_e(t)$ via:

$$E[\xi_k] = \frac{S_{\text{total}} \cdot \mu}{k} \cdot \int_0^\infty \binom{n}{k} \frac{t^{k-1}}{N_e(t)^{n-1}} \cdot (\text{coalescent density terms}) \, dt$$

In practice, we use a fast SFS-based estimator (stairwayplot2-style or momi2-style) to fit $N_e(t)$ in $M$ epochs. This is a one-time computation taking seconds.

**GPU implementation:** Computing the SFS is a column-wise popcount of the genotype matrix — one CUDA kernel, $O(n \cdot S)$ work, trivially parallel. The $N_e(t)$ fitting is a small optimization problem ($M \approx 20$ parameters) done on CPU.

### 1.4 Across-Pair Ultrametric Consistency

At any genomic site $s$, the true pairwise TMRCAs for any subset of $m$ haplotypes must satisfy the **ultrametric constraint**: for any triple $(i, j, k)$, the two largest values among $\{T_{ij}, T_{ik}, T_{jk}\}$ must be equal. This is because the pairwise TMRCAs are determined by a bifurcating tree.

Given posterior marginals from the per-pair HMMs $P(T_{ij}(s) = \bar{t}_k \mid \text{data})$ for all pairs in the subsample, we seek the ultrametric-consistent assignment that maximizes the joint posterior:

$$\hat{\boldsymbol{\tau}}(s) = \arg\max_{\boldsymbol{\tau} \in \mathcal{U}_m} \sum_{(i,j)} \log P(T_{ij}(s) = \tau_{ij} \mid \text{data})$$

where $\mathcal{U}_m$ is the set of $m \times m$ ultrametric matrices with entries drawn from $\{\bar{t}_1, \ldots, \bar{t}_K\}$.

**Solution:** This can be solved exactly via a modified agglomerative clustering:

1. Initialize: each haplotype is its own cluster
2. At each merge step, find the pair of clusters $(A, B)$ that maximizes the sum of log-posteriors when assigned the merge-time $\bar{t}_k$:
   $$\text{score}(A, B, k) = \sum_{i \in A, j \in B} \log P(T_{ij}(s) = \bar{t}_k \mid \text{data})$$
3. Merge the pair with the highest score across all $k$, at the optimal time bin
4. Repeat until one cluster remains

This is $O(m^2 K + m^3)$ per site. For $m = 20$ and $K = 32$, it's ~20K operations — microseconds per site.

### 1.5 Expectation Propagation Loop

The two factors — along-genome HMM and across-pair ultrametric — are coupled via **message passing**:

**Initialization:**
- Set prior messages $\mathbf{m}_{ij}(s) = \mathbf{q}$ (the coalescent prior) for all pairs and sites

**Iteration:**
1. **Along-genome pass** (parallel over pairs): For each pair $(i,j)$, run forward-backward with emission likelihoods and transition matrix, but with the prior at each site replaced by $\mathbf{m}_{ij}(s)$. This yields posterior marginals $\boldsymbol{\gamma}_{ij}(s) = P(T_{ij}(s) = \bar{t}_k \mid \text{data}, \mathbf{m})$.

2. **Across-pair pass** (parallel over sites): For each site $s$, subsample $m$ haplotypes. Use their posteriors $\boldsymbol{\gamma}_{ij}(s)$ to find the ultrametric-consistent MAP assignment $\hat{\boldsymbol{\tau}}(s)$. Update messages:
   $$m_{ij,k}(s) \propto \begin{cases} \alpha \cdot \delta(k = \hat{\tau}_{ij}(s)) + (1 - \alpha) \cdot q_k & \text{if } (i,j) \text{ in subsample} \\ m_{ij,k}(s) & \text{otherwise} \end{cases}$$
   where $\alpha \in (0, 1)$ is a damping parameter (default $\alpha = 0.5$) that controls how strongly the tree constraint influences the HMM.

3. **Convergence check:** Compute $\Delta = \max_{ij,s} |\boldsymbol{\gamma}_{ij}^{(t)}(s) - \boldsymbol{\gamma}_{ij}^{(t-1)}(s)|_1$. Stop if $\Delta < \epsilon$ (default $\epsilon = 0.01$).

**Expected convergence:** 3–5 iterations based on analogous EP schemes.

---

## 2. Three-Tier Architecture

The system is designed as three tiers of increasing accuracy and computational cost. All tiers share the same bitpacked genotype representation on GPU and can be composed.

### Tier 1: Instant Divergence (Method of Moments)

**What it computes:** Windowed pairwise divergence $\hat{T}_{ij}(s) = \pi_{ij}(s) / (2\mu)$ at multiple window scales.

**Algorithm:**
1. Bitpack genotype matrix: each haplotype stored as a vector of `uint64` words, each encoding 64 consecutive sites
2. For each pair tile (batch of pairs processed together):
   a. XOR the bitpacked vectors → pairwise difference bitvector
   b. Parallel prefix scan (`__popcll` + scan) → cumulative difference count $C_{ij}[s]$
   c. For each window scale $W$: $\pi_{ij}(s) = C_{ij}[s + W] - C_{ij}[s - W]$
   d. Divide by $2\mu \cdot 2W$ to get TMRCA estimate
3. Accumulate per-site summary statistics (mean, variance) across pair tiles

**Use case:** Global $\pi(s)$ landscape, rough TMRCA for all pairs, input to Tier 2.

**Accuracy:** CV ≈ 50–60% per pair per site. Excellent for genome-wide averages.

### Tier 2: Changepoint Segmentation (Maximum Likelihood)

**What it computes:** Piecewise-constant TMRCA segments per pair via changepoint detection.

**Algorithm (PELT — Pruned Exact Linear Time):**

For each pair $(i,j)$, the XOR bitvector $\mathbf{d}_{ij}$ is a binary sequence. We seek the segmentation that minimizes the negative Poisson log-likelihood plus a penalty:

$$\min_{\text{changepoints } \mathcal{B}} \sum_{\text{segments}} \left[ -C_{\text{seg}} \log \hat{\lambda}_{\text{seg}} + \hat{\lambda}_{\text{seg}} \cdot L_{\text{seg}} \right] + |\mathcal{B}| \cdot \beta$$

where $C_{\text{seg}}$ = mutation count in segment, $L_{\text{seg}}$ = physical length, $\hat{\lambda}_{\text{seg}} = C_{\text{seg}} / L_{\text{seg}}$ = MLE rate, and $\beta$ is the BIC penalty ($\beta = \log(S_{\text{total}})$).

The key PELT insight: segments whose cost exceeds the best cost plus penalty can be pruned, giving $O(S)$ expected time.

**GPU implementation:** Each pair runs an independent PELT instance. The inner loop at each site $s$ iterates over non-pruned candidate changepoints and computes segment costs using the prefix-sum array. This is a sequential scan per pair but parallel across pairs — assign one warp (32 threads) per pair, with threads collaborating on the candidate changepoint evaluation via warp-level reductions.

**Output:** Per-pair list of segments with boundaries and MLE TMRCA estimates. This feeds into Tier 3 as initialization.

**Accuracy:** Statistically efficient (MLE) within correctly identified segments. CV ≈ 20–30%. Segment boundaries may be slightly misplaced.

### Tier 3: Variational SMC Inference (Posterior with Tree Constraints)

**What it computes:** Posterior mean and credible intervals of $T_{ij}(s)$ via the EP loop described in §1.5.

**Algorithm:** See §1.5. The core computational kernels are:

1. **Batched HMM forward-backward** (§3.1)
2. **Batched ultrametric projection** (§3.2)
3. **Message update** (§3.3)

**Accuracy:** Comparable to MSMC2. CV ≈ 10–15%. Proper uncertainty quantification via posterior width.

---

## 3. CUDA Kernel Specifications

### 3.0 Data Structures

All data structures live in GPU global memory (HBM) unless noted.

```c
// === Core genotype data ===
typedef struct {
    uint64_t* packed;       // Bitpacked genotype matrix [n × n_words]
                            // n_words = ceil(S / 64)
                            // Bit j of packed[i * n_words + w] = G[i][64*w + j]
    double* positions;      // Physical positions of segregating sites [S]
    int n;                  // Number of haplotypes
    int S;                  // Number of segregating sites
    int n_words;            // Number of uint64 words per haplotype
} GenotypeMatrix;

// === Recombination and mutation maps ===
typedef struct {
    double* mu;             // Per-site mutation rate [S]
    double* rho;            // Per-site recombination rate [S] (rate to next site)
    double* cum_rho;        // Cumulative recombination distance [S]
                            // cum_rho[s] = sum_{i=0}^{s-1} rho[i] * (pos[i+1] - pos[i])
} GeneticMaps;

// === Demographic model ===
typedef struct {
    double* Ne;             // Piecewise-constant Ne values [M]
    double* epoch_boundaries; // Time boundaries for Ne epochs [M+1]
    int M;                  // Number of epochs
    
    // Precomputed from Ne:
    double* coal_prior;     // Coalescent prior q[k] for each time bin [K]
    double* cum_coal_rate;  // Cumulative coalescent rate integral [K]
} Demography;

// === HMM parameters (precomputed, read-only) ===
#define K 32  // Number of discrete time bins

typedef struct {
    double time_midpoints[K];     // Midpoint of each time bin
    double time_boundaries[K+1];  // Boundaries of time bins
    
    // Transition matrices are position-dependent (depend on recombination distance)
    // so they are computed on-the-fly in the kernel from:
    //   A[k][l](r) = exp(-r * t_k) * delta(k,l) + (1 - exp(-r * t_k)) * q[l]
    // We store the coalescent prior q[k] and compute A on the fly.
    
    double coal_prior[K];         // q[k] = prior probability of coalescing in bin k
} HMMParams;

// === Per-pair HMM state ===
typedef struct {
    // Forward-backward arrays: [n_pairs_in_tile × S × K]
    // Stored in half-precision (float16) to save memory
    __half* alpha;          // Forward probabilities
    __half* beta;           // Backward probabilities
    __half* gamma;          // Posterior marginals P(T=t_k | data)
} HMMState;

// === Message arrays for EP ===
typedef struct {
    // Prior messages from tree constraint: [n_pairs_in_tile × S × K]
    __half* messages;       // m_ij(s) = modified prior at each site
    __half* messages_prev;  // Previous iteration (for convergence check)
} EPMessages;

// === Output ===
typedef struct {
    // Per-pair per-site posterior summary: [n_target_pairs × S]
    float* tmrca_mean;      // E[T_ij(s) | data]
    float* tmrca_lower;     // 2.5th percentile
    float* tmrca_upper;     // 97.5th percentile
    
    // Per-site summary across all pairs: [S]
    float* pi;              // Mean pairwise divergence
    float* tmrca_site_mean; // Mean TMRCA across pairs
    float* tmrca_site_var;  // Variance of TMRCA across pairs
} Output;

// === Pair tiling ===
#define PAIR_TILE_SIZE 4096   // Number of pairs processed simultaneously
// For n=500K, total pairs ≈ 1.25e11
// Tile count ≈ 3e7
// Memory per tile: PAIR_TILE_SIZE × S × K × sizeof(__half) 
//   = 4096 × 1e7 × 32 × 2 bytes ≈ 2.6 TB (TOO LARGE)
//
// SOLUTION: Process sites in blocks too.
#define SITE_BLOCK_SIZE 65536  // ~65K sites per block
// Memory per tile-block: 4096 × 65536 × 32 × 2 ≈ 16 GB (fits on one A100)

// Pair indexing: pair index p maps to (i,j) via:
//   i = floor((1 + sqrt(1 + 8p)) / 2)
//   j = p - i*(i-1)/2
// This avoids storing explicit pair lists.
```

### 3.1 Kernel: Bitpack and Prefix Scan

**Purpose:** Transform raw genotype matrix into bitpacked representation and precompute cumulative difference counts for all pairs in a tile.

```
Kernel: bitpack_genotypes
Grid: ceil(n / 256) blocks × ceil(n_words / 1) blocks  (2D grid)
Block: 256 threads
Input: Raw genotype matrix G[n][S] (uint8 or bool)
Output: GenotypeMatrix.packed[n][n_words] (uint64)

Algorithm:
  Each thread handles one haplotype × one word:
    word = 0
    for bit in 0..63:
      site = block_word * 64 + bit
      if site < S:
        word |= ((uint64_t)G[haplotype][site]) << bit
    packed[haplotype][block_word] = word
```

```
Kernel: pairwise_prefix_scan
Grid: ceil(n_pairs_in_tile / 32) blocks  (one warp per pair)
Block: 32 threads (one warp)
Input: packed[n][n_words], pair_indices[PAIR_TILE_SIZE][2]
Output: prefix[PAIR_TILE_SIZE][n_words] (cumulative popcount)

Algorithm per pair (i,j), processed by one warp:
  running_sum = 0
  for w in 0..n_words-1 (warp collaborates via striped access):
    xor_word = packed[i][w] ^ packed[j][w]
    count = __popcll(xor_word)
    // Warp-level inclusive prefix scan of counts across words
    // Each thread handles n_words/32 consecutive words
    running_sum += count
    prefix[pair_idx][w] = running_sum
```

Note: For Tier 1 only, the prefix scan output is used directly. For Tiers 2–3, the prefix scan is an intermediate that feeds the HMM.

### 3.2 Kernel: SFS Computation (for Demography Estimation)

**Purpose:** Compute the folded/unfolded site frequency spectrum from the genotype matrix.

```
Kernel: compute_sfs
Grid: ceil(S / 256) blocks
Block: 256 threads
Input: packed[n][n_words]
Output: sfs[n+1] (allele count histogram)

Algorithm:
  Each thread handles one site:
    site_word = site / 64
    site_bit = site % 64
    count = 0
    for i in 0..n-1:
      count += (packed[i][site_word] >> site_bit) & 1
    atomicAdd(&sfs[count], 1)
```

For large $n$, the inner loop is expensive. Optimization: transpose the bitpacked matrix so that all haplotypes for one word are contiguous, then use warp-level `__popcll` reduction across haplotypes. This gives $O(n/64)$ operations per site instead of $O(n)$.

### 3.3 Kernel: Batched HMM Forward-Backward

**This is the most critical kernel in the system.**

**Purpose:** Run the SMC forward-backward algorithm simultaneously for all pairs in a tile, producing posterior marginals $\gamma_{ij}(s, k) = P(T_{ij}(s) = \bar{t}_k \mid \text{data}, \text{messages})$.

```
Kernel: hmm_forward_backward
Grid: PAIR_TILE_SIZE blocks  (one block per pair)
Block: K threads = 32 threads  (one thread per time bin)
Shared memory: 3 × K doubles = 768 bytes (alpha, emission, transition_col)

Input:
  - packed[n][n_words]: bitpacked genotypes
  - positions[S]: site positions
  - genetic_maps: mu[S], cum_rho[S]
  - hmm_params: time_midpoints[K], coal_prior[K]
  - messages[PAIR_TILE_SIZE][S][K]: prior messages from EP (or coal_prior if first iter)
  - pair_indices[PAIR_TILE_SIZE][2]: (i,j) pair for each slot

Output:
  - gamma[PAIR_TILE_SIZE][S][K]: posterior marginals (__half precision)
  - log_likelihood[PAIR_TILE_SIZE]: total log-likelihood per pair (for convergence)

Algorithm (one block = one pair):

  thread_k = threadIdx.x   // This thread handles time bin k
  i, j = pair_indices[blockIdx.x]
  
  // === FORWARD PASS ===
  
  // Initialize alpha[0] with prior (message at site 0)
  alpha[thread_k] = messages[blockIdx.x][0][thread_k]
  
  // Emission at site 0
  d = get_xor_bit(packed, i, j, 0)  // XOR bit at site 0
  mu_s = mu[0]
  t_k = time_midpoints[thread_k]
  if d == 1:
    emit = 1.0 - exp(-2.0 * mu_s * t_k)
  else:
    emit = exp(-2.0 * mu_s * t_k)
  alpha[thread_k] *= emit
  
  // Normalize
  __syncthreads()
  sum = warp_reduce_sum(alpha[thread_k])  // Warp-level reduction
  alpha[thread_k] /= sum
  log_lik = log(sum)
  
  // Store gamma for site 0 (will be updated in backward pass)
  gamma[blockIdx.x][0][thread_k] = __float2half(alpha[thread_k])
  
  for s = 1 to S-1:
    __syncthreads()
    
    // --- Transition ---
    // Recombination distance between site s-1 and s
    r = cum_rho[s] - cum_rho[s-1]
    
    // Gap emission: probability of no mutations in the gap between sites
    gap_bp = positions[s] - positions[s-1] - 1
    if gap_bp > 0:
      mu_gap = (mu[s] + mu[s-1]) / 2.0  // average rate in gap
      gap_emit = exp(-2.0 * mu_gap * t_k * gap_bp)
    else:
      gap_emit = 1.0
    
    // Transition: alpha_new[k] = sum_l A[l][k](r) * alpha[l]
    // A[l][k](r) = exp(-r * t_l) * delta(l,k) + (1 - exp(-r * t_l)) * q[k]
    //
    // This can be decomposed:
    //   alpha_new[k] = exp(-r * t_k) * alpha[k]          (stay in same bin)
    //               + q[k] * sum_l (1 - exp(-r * t_l)) * alpha[l]  (recombine)
    //
    // The second term has the same sum for all k, so compute it once:
    
    stay = exp(-r * t_k) * alpha[thread_k]
    recomb_mass = (1.0 - exp(-r * t_k)) * alpha[thread_k]
    
    __syncthreads()
    total_recomb = warp_reduce_sum(recomb_mass)  // sum_l (1-exp(-r*t_l))*alpha[l]
    
    alpha_new = stay + coal_prior[thread_k] * total_recomb
    
    // Incorporate message (prior from EP tree constraint)
    alpha_new *= messages[blockIdx.x][s][thread_k]
    
    // Gap emission
    alpha_new *= gap_emit
    
    // Site emission
    d = get_xor_bit(packed, i, j, s)
    mu_s = mu[s]
    if d == 1:
      emit = 1.0 - exp(-2.0 * mu_s * t_k)
    else:
      emit = exp(-2.0 * mu_s * t_k)
    alpha_new *= emit
    
    // Normalize
    __syncthreads()
    sum = warp_reduce_sum(alpha_new)
    alpha_new /= sum
    log_lik += log(sum)
    
    alpha[thread_k] = alpha_new
    gamma[blockIdx.x][s][thread_k] = __float2half(alpha_new)  // temporary; updated in backward
  
  // === BACKWARD PASS ===
  
  beta[thread_k] = 1.0  // Initialize at last site
  
  // Update gamma at last site
  gamma_val = alpha[thread_k] * beta[thread_k]
  // (already normalized since beta=1 and alpha is normalized)
  gamma[blockIdx.x][S-1][thread_k] = __float2half(gamma_val)
  
  for s = S-2 downto 0:
    __syncthreads()
    
    // Reverse transition (transpose of forward transition)
    r = cum_rho[s+1] - cum_rho[s]
    
    // beta_new[k] = sum_l A[k][l](r) * emit[l](s+1) * gap_emit[l] * beta[l]
    //
    // Using same decomposition:
    d_next = get_xor_bit(packed, i, j, s+1)
    mu_next = mu[s+1]
    t_k = time_midpoints[thread_k]
    
    if d_next == 1:
      emit_next = 1.0 - exp(-2.0 * mu_next * t_k)
    else:
      emit_next = exp(-2.0 * mu_next * t_k)
    
    gap_bp = positions[s+1] - positions[s] - 1
    mu_gap = (mu[s] + mu[s+1]) / 2.0
    gap_emit = (gap_bp > 0) ? exp(-2.0 * mu_gap * t_k * gap_bp) : 1.0
    
    // Include message at site s+1
    msg = messages[blockIdx.x][s+1][thread_k]
    
    be = beta[thread_k] * emit_next * gap_emit * msg
    
    stay_term = exp(-r * t_k) * be
    recomb_term = (1.0 - exp(-r * t_k)) * be
    
    __syncthreads()
    total_recomb = warp_reduce_sum(recomb_term)
    
    beta_new = stay_term + coal_prior[thread_k] * total_recomb
    
    // Normalize
    __syncthreads()
    sum = warp_reduce_sum(beta_new)
    beta_new /= sum
    
    beta[thread_k] = beta_new
    
    // Compute gamma = alpha * beta, normalized
    // We stored alpha in gamma during forward pass; now update
    alpha_s = __half2float(gamma[blockIdx.x][s][thread_k])
    gamma_val = alpha_s * beta_new
    __syncthreads()
    gamma_sum = warp_reduce_sum(gamma_val)
    gamma_val /= gamma_sum
    
    gamma[blockIdx.x][s][thread_k] = __float2half(gamma_val)
  
  // Store log-likelihood
  if thread_k == 0:
    log_likelihood[blockIdx.x] = log_lik
```

**Critical implementation notes:**
- `get_xor_bit(packed, i, j, s)`: extracts bit `s % 64` from `packed[i][s/64] ^ packed[j][s/64]`. This is 2 global memory reads + XOR + shift. Cache the current words in registers since consecutive sites share the same word.
- The forward and backward passes must iterate over sites sequentially (inherent serial dependency along the genome). However, all pairs in the tile run simultaneously — this is where the parallelism comes from.
- Memory for alpha/beta: only need current and previous site, so $O(K)$ per pair, stored in registers/shared memory. The gamma array is $O(S \times K)$ per pair in global memory.
- For the site-block variant (SITE_BLOCK_SIZE < S): run forward pass on block 0, store final alpha at boundary, then start block 1 from that alpha. Backward pass in reverse. This adds boundary synchronization but keeps memory bounded.

### 3.4 Kernel: Ultrametric Projection

**Purpose:** At each site, find the tree-consistent TMRCA assignment that maximizes the joint posterior across a subsample of haplotypes.

```
Kernel: ultrametric_project
Grid: ceil(S / 1) blocks  (one block per site)
Block: 256 threads
Shared memory: m*(m-1)/2 * K * sizeof(__half) for pairwise posteriors
               + m * sizeof(int) for cluster assignments
               + clustering workspace

Input:
  - gamma[PAIR_TILE_SIZE][SITE_BLOCK_SIZE][K]: posterior marginals
  - subsample_indices[m]: which haplotypes are in the subsample
  - subsample_pair_map[m*(m-1)/2]: maps subsample pair index to pair_tile index
Output:
  - updated_messages[PAIR_TILE_SIZE][SITE_BLOCK_SIZE][K]

Parameters:
  m = 20 (subsample size, configurable)
  n_subsamples = 5 (number of random subsamples, configurable)
  alpha = 0.5 (damping factor)

Algorithm (one block = one site):

  s = blockIdx.x
  
  // Load pairwise posteriors for subsample into shared memory
  // m*(m-1)/2 = 190 pairs for m=20
  for each subsample pair (a,b), a<b:
    pair_idx = subsample_pair_map[pair_index(a,b)]
    for k = 0..K-1 (threaded):
      shm_posterior[pair_index(a,b)][k] = gamma[pair_idx][s][k]
  __syncthreads()
  
  // === Agglomerative clustering with posterior scoring ===
  
  // Initialize: m clusters, each containing one haplotype
  // cluster[i] = i for all i
  // active[i] = true for all i
  
  for merge_step = 0 to m-2:
    best_score = -INF
    best_A = -1, best_B = -1, best_k = -1
    
    // Find best pair of clusters to merge and optimal time bin
    // Threads cooperate: each thread evaluates a subset of (A,B,k) triples
    for each pair of active clusters (A, B):
      for k = 0..K-1:
        score = 0
        for each (i in A, j in B):
          score += log(shm_posterior[pair_index(i,j)][k])
        if score > best_score:
          best_score = score
          best_A = A, best_B = B, best_k = k
    
    // Merge clusters A and B at time bin best_k
    // All pairs (i,j) with i in A, j in B get assigned TMRCA = t_{best_k}
    merge(A, B)
    assigned_time[A][B] = best_k
  
  // === Update messages ===
  for each subsample pair (a,b):
    pair_idx = subsample_pair_map[pair_index(a,b)]
    assigned_k = get_assigned_time(a, b)  // from clustering
    
    for k = 0..K-1:
      // Soft assignment: concentrated on assigned bin but with damping
      tree_msg = alpha * ((k == assigned_k) ? 1.0 : 0.0) + (1.0 - alpha) * coal_prior[k]
      // Blend with previous message (momentum)
      old_msg = messages[pair_idx][s][k]
      new_msg = 0.7 * tree_msg + 0.3 * old_msg  // momentum for stability
      updated_messages[pair_idx][s][k] = __float2half(new_msg)
  
  // Non-subsampled pairs: messages unchanged
```

**Optimization notes:**
- For $m = 20$, the clustering has only 19 merge steps, each evaluating at most $\binom{20}{2} \times K = 190 \times 32 = 6080$ candidates. This is tiny — a single warp can handle it.
- Multiple subsamples ($n_{\text{sub}} = 5$): run sequentially within the block, average the assigned time bins across subsamples before updating messages.
- The subsample should be re-drawn each EP iteration for better coverage.

### 3.5 Kernel: Message Update and Convergence Check

```
Kernel: check_convergence
Grid: standard reduction grid
Block: 256 threads
Input: messages[...][K], messages_prev[...][K]
Output: max_delta (scalar)

Algorithm:
  Standard max-reduction of |messages - messages_prev| across all entries.
  Use warp shuffle + shared memory two-level reduction.
```

### 3.6 Kernel: Posterior Summary Extraction

```
Kernel: extract_summaries
Grid: ceil(n_target_pairs * S / 256) blocks
Block: 256 threads
Input: gamma[...][S][K], time_midpoints[K]
Output: tmrca_mean[...][S], tmrca_lower[...][S], tmrca_upper[...][S]

Algorithm per (pair, site):
  // Posterior mean
  mean = sum_k gamma[k] * time_midpoints[k]
  
  // Credible interval via cumulative posterior
  cum = 0
  lower = time_midpoints[0]
  upper = time_midpoints[K-1]
  for k = 0..K-1:
    cum += gamma[k]
    if cum >= 0.025 and lower == time_midpoints[0]:
      lower = time_midpoints[k]
    if cum >= 0.975:
      upper = time_midpoints[k]
      break
  
  tmrca_mean[pair][s] = mean
  tmrca_lower[pair][s] = lower
  tmrca_upper[pair][s] = upper
```

---

## 4. Memory Management and Streaming Strategy

### 4.1 Memory Budget (Single A100, 80 GB HBM)

| Component | Size | Notes |
|-----------|------|-------|
| Bitpacked genotypes | $n \times S / 8$ bytes | 500K × 10M / 8 = 625 GB — **multi-GPU** |
| Genetic maps | $3 \times S \times 8$ bytes | 240 MB |
| HMM params | $O(K)$ | negligible |
| Per-tile gamma | $T \times B \times K \times 2$ bytes | 4096 × 65536 × 32 × 2 = 16 GB |
| Per-tile messages | same as gamma | 16 GB |
| Per-tile messages_prev | same | 16 GB |
| Working space | ~4 GB | alpha/beta in registers, shared mem |
| **Total per GPU** | **~52 GB + genotype share** | fits on 80 GB A100 |

### 4.2 Multi-GPU Data Distribution

For biobank-scale ($n = 500K$):

**Strategy: Haplotype-sharded genotype matrix.**
- Partition haplotypes across GPUs: GPU $g$ holds haplotypes $[g \cdot n/G, (g+1) \cdot n/G)$
- For pairs $(i,j)$ where both haplotypes are on the same GPU: process locally
- For cross-GPU pairs: use NVLink/NVSwitch peer-to-peer access to read the remote haplotype, or pre-scatter required haplotype tiles

**Communication pattern:** For a pair tile, at most 2 cache lines of haplotype data need to be read per pair per site. With NVLink bandwidth of 600 GB/s, this is not the bottleneck.

**Alternative for very large $n$:** Keep genotypes in CPU RAM (host memory), stream haplotype tiles to GPU via PCIe as needed. PCIe 4.0 x16 = 32 GB/s → reading one pair tile's genotypes (4096 × 156K words × 8 bytes = 5 GB) takes ~150 ms. This is slow but amortizable over the $O(S)$ HMM computation.

### 4.3 Streaming Execution Plan

```python
# Pseudocode for the full pipeline orchestration

def run_gamma_smc_cu(G, positions, mu, rho, target_pairs=None, 
                 tier=3, m_subsample=20, n_subsamples=5,
                 max_ep_iter=5, damping=0.5, convergence_tol=0.01):
    
    # ============================================
    # Phase 0: Preprocessing (CPU + one GPU pass)
    # ============================================
    
    G_packed = bitpack_genotypes(G)              # GPU kernel
    cum_rho = cumulative_sum(rho * diff(positions))  # CPU or GPU scan
    
    # Compute SFS and estimate demography
    sfs = compute_sfs(G_packed)                  # GPU kernel
    Ne_history = fit_demography_from_sfs(sfs)    # CPU optimization
    coal_prior = compute_coalescent_prior(Ne_history, time_bins)  # CPU
    
    # Determine pairs to process
    if target_pairs is None:
        if tier <= 2:
            # All pairs, streamed
            target_pairs = "all"
        else:
            # Subsample for Tier 3
            target_pairs = random_subsample_pairs(n, m_subsample)
    
    n_pairs = len(target_pairs)  # or n*(n-1)/2 for "all"
    
    # ============================================
    # Phase 1: Tier 1 — Instant divergence
    # ============================================
    
    if tier >= 1:
        pi_per_site = zeros(S)
        
        for pair_tile in tile_pairs(target_pairs, PAIR_TILE_SIZE):
            # Compute prefix scans for this tile
            prefix = pairwise_prefix_scan(G_packed, pair_tile)
            
            # Windowed divergence at multiple scales
            for W in [1000, 10000, 100000]:
                div = windowed_divergence(prefix, positions, W)
                # Accumulate into per-site statistics
                accumulate_site_stats(div, pi_per_site)
            
            # Free prefix (recomputed if needed later)
        
        if tier == 1:
            return pi_per_site / (2 * mu)
    
    # ============================================
    # Phase 2: Tier 2 — Changepoint segmentation  
    # ============================================
    
    if tier >= 2:
        segments = {}  # pair -> list of (start, end, tmrca_mle)
        
        for pair_tile in tile_pairs(target_pairs, PAIR_TILE_SIZE):
            prefix = pairwise_prefix_scan(G_packed, pair_tile)
            tile_segments = pelt_changepoint(prefix, positions, mu)
            segments.update(tile_segments)
        
        if tier == 2:
            return segments
    
    # ============================================
    # Phase 3: Tier 3 — Variational SMC inference
    # ============================================
    
    # Initialize messages to coalescent prior
    # (messages are stored per site-block, streamed)
    messages = initialize_messages(coal_prior, n_pairs, S)
    
    for ep_iter in range(max_ep_iter):
        max_delta = 0
        
        # --- Along-genome pass ---
        for pair_tile in tile_pairs(target_pairs, PAIR_TILE_SIZE):
            for site_block in tile_sites(S, SITE_BLOCK_SIZE):
                
                # Load genotype words for this tile × block
                load_genotype_tile(G_packed, pair_tile, site_block)
                
                # Load messages for this tile × block
                load_messages(messages, pair_tile, site_block)
                
                # Run HMM forward-backward
                gamma = hmm_forward_backward(
                    G_packed, pair_tile, site_block,
                    positions, mu, cum_rho,
                    coal_prior, messages
                )
                
                # Store gamma for use in across-pair pass
                store_gamma(gamma, pair_tile, site_block)
        
        # --- Across-pair pass ---
        for site_block in tile_sites(S, SITE_BLOCK_SIZE):
            # Draw random subsamples
            subsamples = draw_subsamples(n_pairs, m_subsample, n_subsamples)
            
            # Load gamma for relevant pairs in this site block
            gamma_block = load_gamma_for_subsamples(subsamples, site_block)
            
            # Run ultrametric projection
            new_messages = ultrametric_project(
                gamma_block, subsamples, coal_prior, damping
            )
            
            # Compute convergence metric
            delta = max_abs_diff(new_messages, messages[site_block])
            max_delta = max(max_delta, delta)
            
            # Update messages
            messages[site_block] = new_messages
        
        print(f"EP iteration {ep_iter}: max_delta = {max_delta:.6f}")
        if max_delta < convergence_tol:
            break
    
    # ============================================
    # Phase 4: Extract posteriors
    # ============================================
    
    results = extract_summaries(gamma, time_midpoints)
    return results
```

### 4.4 Site-Block Boundary Handling for HMM

The HMM has a sequential dependency along sites. When processing in site blocks, we must handle boundaries:

```
Site block processing:

Block 0: sites [0, B)
  Forward pass: alpha starts from prior at site 0
  Store alpha[B-1] as "boundary alpha" for block 1
  Backward pass: beta starts from 1.0 at site B-1 (WRONG — needs info from block 1)

Solution: Two-pass approach:
  Pass 1 (Forward): Process all blocks left-to-right, storing boundary alphas
  Pass 2 (Backward): Process all blocks right-to-left, using boundary betas
  
  Between passes, only boundary vectors (K floats per pair) need to be stored.
  Boundary storage: PAIR_TILE_SIZE × ceil(S/SITE_BLOCK_SIZE) × K × 4 bytes
    = 4096 × 154 × 32 × 4 = 77 MB (negligible)
```

---

## 5. Python API and Integration

### 5.1 Python API

```python
import gamma_smc_cu

# ============================================
# Initialize from various input formats
# ============================================

# From numpy array
fc = gamma_smc_cu.CoalescenceEstimator(
    genotypes=np.array(..., dtype=np.uint8),  # (n, S) haploid genotype matrix
    positions=np.array(..., dtype=np.float64), # (S,) physical positions in bp
    mu=1.25e-8,               # scalar or array of per-site rates
    rho=1e-8,                 # scalar or array of per-site recomb rates
    gpu_ids=[0, 1, 2, 3],    # GPUs to use
)

# From tskit TreeSequence (for validation against msprime simulations)
fc = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts, gpu_ids=[0])

# From VCF (streams variants, bitpacks on the fly)
fc = gamma_smc_cu.CoalescenceEstimator.from_vcf("chr1.vcf.gz", gpu_ids=[0,1,2,3])

# From PLINK bed/bim/fam
fc = gamma_smc_cu.CoalescenceEstimator.from_plink("biobank", gpu_ids=range(8))

# ============================================
# Tier 1: Instant divergence
# ============================================

pi = fc.site_pi()  
# Returns: np.ndarray of shape (S,) — per-site nucleotide diversity

tmrca_rough = fc.pairwise_divergence(
    pairs=[(0,1), (2,3)],  # specific pairs, or "all"
    window_sizes=[1000, 10000, 100000],
)
# Returns: np.ndarray of shape (n_pairs, S, n_windows)

# ============================================
# Tier 2: Changepoint segmentation
# ============================================

segments = fc.segment_tmrca(pairs="all")
# Returns: dict mapping (i,j) -> list of Segment(start, end, tmrca, n_mutations)

# ============================================
# Tier 3: Variational SMC inference
# ============================================

result = fc.infer_tmrca(
    pairs="subsample",        # or list of specific pairs, or "all"
    subsample_size=200,       # m: haplotypes in subsample
    n_subsamples=5,           # random subsamples for ultrametric step
    max_iterations=5,         # EP iterations
    n_time_bins=32,           # K: discretization
    damping=0.5,              # EP damping
    t_max=None,               # auto-determined from demography
)

# result.tmrca_mean: (n_pairs, S) posterior mean TMRCA
# result.tmrca_lower: (n_pairs, S) 2.5th percentile
# result.tmrca_upper: (n_pairs, S) 97.5th percentile  
# result.demography: fitted Ne(t) history
# result.log_likelihood: per-pair total log-likelihood
# result.segments: Tier 2 segments (computed as initialization)
# result.converged: bool

# ============================================
# Convenience methods
# ============================================

# Genome-wide TMRCA landscape (averaged over pairs)
landscape = fc.tmrca_landscape(n_subsample=200, resolution=1000)
# Returns: (n_bins,) mean TMRCA in non-overlapping windows

# Cross-population coalescence times
cross_tmrca = fc.cross_population_tmrca(
    pop1_indices=[0, 1, 2, ...],
    pop2_indices=[100, 101, 102, ...],
    n_pairs=1000,
)

# IBD segment detection (thresholding Tier 3 output)
ibd = fc.detect_ibd(pair=(0, 1), tmrca_threshold=100)
# Returns: list of IBDSegment(start, end, tmrca, length_cM)
```

### 5.2 Validation Against msprime / tskit

```python
import msprime
import gamma_smc_cu
import numpy as np

def validate_accuracy():
    """
    Simulate under known demography, compare gamma_smc_cu TMRCA estimates
    to true TMRCA from the tree sequence.
    """
    # Simulate with known demography
    demography = msprime.Demography()
    demography.add_population(initial_size=10000)
    demography.add_population_parameters_change(time=5000, initial_size=1000)  # bottleneck
    demography.add_population_parameters_change(time=6000, initial_size=50000)  # expansion
    
    ts = msprime.sim_ancestry(
        samples=100,  # 100 diploid = 200 haploid
        sequence_length=10_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
    
    # True pairwise TMRCA at each tree
    true_tmrca = {}
    for pair in [(0,1), (0,2), (1,2), (0,50), (0,100)]:
        i, j = pair
        tmrca_array = np.zeros(int(ts.sequence_length))
        for tree in ts.trees():
            t = tree.tmrca(i, j)
            left, right = int(tree.interval.left), int(tree.interval.right)
            tmrca_array[left:right] = t
        true_tmrca[pair] = tmrca_array
    
    # gamma_smc_cu estimates
    fc = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts, gpu_ids=[0])
    result = fc.infer_tmrca(
        pairs=list(true_tmrca.keys()),
        subsample_size=50,
        max_iterations=5,
    )
    
    # Compare
    for idx, pair in enumerate(true_tmrca.keys()):
        true_t = true_tmrca[pair]
        est_t = result.tmrca_mean[idx]
        
        # Interpolate estimated TMRCA (defined at segregating sites) 
        # to all base positions
        est_interp = np.interp(
            np.arange(len(true_t)),
            fc.positions,
            est_t
        )
        
        # Metrics
        correlation = np.corrcoef(true_t, est_interp)[0, 1]
        rmse = np.sqrt(np.mean((true_t - est_interp)**2))
        relative_error = np.mean(np.abs(true_t - est_interp) / (true_t + 1))
        
        print(f"Pair {pair}: r={correlation:.4f}, RMSE={rmse:.0f}, "
              f"rel_error={relative_error:.3f}")
    
    # Expected: r > 0.85, relative_error < 0.20 for Tier 3
    # Compare against MSMC2 and PSMC on same data for benchmarking
```

---

## 6. Build System and Dependencies

### 6.1 Project Structure

```
gamma_smc_cu/
├── CMakeLists.txt                   # Top-level CMake
├── setup.py                         # Python package setup (scikit-build or meson)
├── pyproject.toml
│
├── include/
│   └── gamma_smc_cu/
│       ├── types.h                  # Core data structures (§3.0)
│       ├── bitpack.h                # Bitpacking utilities
│       ├── hmm.h                    # HMM parameters and functions
│       └── api.h                    # C API for Python bindings
│
├── src/
│   ├── kernels/
│   │   ├── bitpack.cu               # Kernel §3.1: bitpack + prefix scan
│   │   ├── sfs.cu                   # Kernel §3.2: SFS computation
│   │   ├── hmm_forward_backward.cu  # Kernel §3.3: batched HMM
│   │   ├── ultrametric.cu           # Kernel §3.4: ultrametric projection
│   │   ├── convergence.cu           # Kernel §3.5: message update + convergence
│   │   ├── summary.cu              # Kernel §3.6: posterior summary extraction
│   │   ├── tier1_divergence.cu      # Tier 1 windowed divergence
│   │   └── tier2_pelt.cu           # Tier 2 PELT changepoint detection
│   │
│   ├── pipeline.cu                  # Orchestration: streaming pair/site tiles
│   ├── demography.cpp               # Ne(t) fitting from SFS (CPU)
│   ├── genetic_map.cpp              # Loading/interpolating genetic maps
│   └── io/
│       ├── vcf_reader.cpp           # VCF streaming input
│       ├── plink_reader.cpp         # PLINK bed/bim/fam input
│       └── tskit_bridge.cpp         # tskit TreeSequence interop
│
├── python/
│   └── gamma_smc_cu/
│       ├── __init__.py
│       ├── estimator.py             # CoalescenceEstimator class (§5.1)
│       ├── _bindings.pyx            # Cython or pybind11 bindings
│       └── plotting.py             # Visualization utilities
│
├── tests/
│   ├── conftest.py                  # Shared fixtures: simulated data, GPU setup
│   ├── unit/
│   │   ├── test_bitpack.py          # Bitpacking correctness
│   │   ├── test_prefix_scan.py      # Prefix scan accuracy
│   │   ├── test_sfs.py              # SFS computation
│   │   ├── test_hmm.py              # HMM forward-backward against numpy ref
│   │   ├── test_hmm_numerical.py    # HMM numerical stability edge cases
│   │   ├── test_ultrametric.py      # Ultrametric projection correctness
│   │   ├── test_pelt.py             # PELT changepoint detection
│   │   ├── test_emissions.py        # Emission probability computations
│   │   ├── test_transitions.py      # Transition matrix construction
│   │   ├── test_demography.py       # Ne(t) fitting from SFS
│   │   ├── test_summary.py          # Posterior summary extraction
│   │   └── test_pair_indexing.py    # Pair ↔ (i,j) index mapping
│   │
│   ├── integration/
│   │   ├── test_tier1_pipeline.py   # End-to-end Tier 1
│   │   ├── test_tier2_pipeline.py   # End-to-end Tier 2
│   │   ├── test_tier3_pipeline.py   # End-to-end Tier 3
│   │   ├── test_ep_convergence.py   # EP loop convergence behavior
│   │   ├── test_site_blocking.py    # HMM site-block boundary handling
│   │   ├── test_pair_tiling.py      # Pair tile streaming correctness
│   │   ├── test_multi_gpu.py        # Multi-GPU sharding consistency
│   │   └── test_io_formats.py       # VCF, PLINK, tskit input round-trips
│   │
│   ├── statistical/
│   │   ├── test_accuracy_msprime.py # Validation against true TMRCAs (§5.2)
│   │   ├── test_calibration.py      # Posterior calibration (coverage)
│   │   ├── test_demography_recovery.py # Ne(t) recovery under known models
│   │   ├── test_demographic_scenarios.py # Bottleneck, expansion, structure
│   │   └── test_mutation_rate_var.py # Variable μ(s) handling
│   │
│   ├── property/
│   │   ├── test_invariants.py       # Exchangeability, symmetry, normalization
│   │   └── test_edge_cases.py       # Monomorphic sites, singletons, no data
│   │
│   ├── regression/
│   │   ├── test_known_outputs.py    # Frozen outputs from validated runs
│   │   └── golden/                  # Golden test data files
│   │       ├── small_sim.npz
│   │       └── expected_posteriors.npz
│   │
│   ├── performance/
│   │   ├── test_scaling.py          # Wall-clock scaling with n and S
│   │   ├── test_memory.py           # GPU memory usage stays within bounds
│   │   └── test_throughput.py       # Sites/second, pairs/second metrics
│   │
│   └── reference/
│       ├── hmm_numpy.py             # Pure numpy HMM for verification
│       ├── ultrametric_numpy.py     # Pure numpy ultrametric for verification
│       ├── pelt_numpy.py            # Pure numpy PELT for verification
│       └── sfs_numpy.py             # Pure numpy SFS for verification
│
└── benchmarks/
    ├── bench_msmc2_comparison.py    # Compare accuracy against MSMC2
    ├── bench_tsdate_comparison.py   # Compare accuracy against tsdate
    ├── bench_relate_comparison.py   # Compare accuracy against Relate
    └── bench_scaling.py             # Wall-clock scaling with n and S
```

### 6.2 Dependencies

**C++/CUDA:**
- CUDA Toolkit >= 11.7 (for cooperative groups, `__popcll`, warp intrinsics)
- CUB library (included in CUDA Toolkit) for device-wide scans and reductions
- htslib (for VCF reading)
- tskit C library (for tree sequence input)

**Python:**
- numpy
- pybind11 or cython (for bindings)
- msprime, tskit (for validation and input)
- cyvcf2 (alternative VCF reader)
- matplotlib (for plotting utilities)

**Build:**
- CMake >= 3.18
- scikit-build or meson-python for Python packaging
- Target architectures: sm_80 (A100), sm_90 (H100)

### 6.3 Build Commands

```bash
# Build C++/CUDA library
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
pip install -e .

# Run tests
pytest tests/ -v

# Run accuracy benchmark
python benchmarks/bench_msmc2_comparison.py
```

---

## 7. Implementation Order and Milestones

### Milestone 1: Core Infrastructure (Week 1–2)

**Goal:** Bitpacking, prefix scan, and Tier 1 divergence working end-to-end.

Tasks:
1. Implement `types.h` with all data structures
2. Implement `bitpack.cu`: genotype bitpacking kernel
3. Implement `bitpack.cu`: pairwise XOR + prefix scan kernel
4. Implement `tier1_divergence.cu`: windowed divergence from prefix sums
5. Write `hmm_numpy.py` reference implementation for later validation
6. Python binding for Tier 1: `fc.site_pi()` and `fc.pairwise_divergence()`
7. Test against msprime: verify $\pi(s)$ matches `ts.diversity()`

**Validation:** Per-site $\pi$ from gamma_smc_cu matches tskit within floating-point tolerance.

### Milestone 2: HMM Forward-Backward (Week 2–4)

**Goal:** Single-pair and batched HMM producing correct posteriors.

Tasks:
1. Implement `demography.cpp`: SFS → $N_e(t)$ fitting
2. Implement `sfs.cu`: GPU SFS computation
3. Implement `hmm_forward_backward.cu`: single-pair forward-backward (debug version with full precision)
4. Validate against `hmm_numpy.py` on simulated data
5. Optimize: half-precision gamma storage, register-resident alpha/beta
6. Implement batched version: one block per pair
7. Implement site-block boundary handling
8. Profile and tune: occupancy, memory throughput, shared memory usage

**Validation:** Per-site posterior means from CUDA kernel match numpy reference to $< 1\%$ relative error. Log-likelihoods match.

### Milestone 3: Tier 2 Changepoint Detection (Week 3–4)

**Goal:** PELT-based segmentation running on GPU.

Tasks:
1. Implement `tier2_pelt.cu`: PELT algorithm, one warp per pair
2. Validate segment boundaries against R `changepoint` package
3. Compare segment-level MLE TMRCAs against true TMRCAs from msprime

**Validation:** Segments capture >80% of true recombination breakpoints within 5 kb. MLE TMRCAs within segments have correlation >0.7 with truth.

### Milestone 4: Ultrametric Projection (Week 4–5)

**Goal:** Tree-consistent TMRCA correction working per site.

Tasks:
1. Implement `ultrametric_numpy.py` reference
2. Implement `ultrametric.cu`: agglomerative clustering kernel
3. Validate: given true posteriors (delta functions at true TMRCAs), verify that ultrametric projection recovers the correct tree
4. Test with noisy posteriors from HMM: verify denoising effect

**Validation:** On simulated data with known tree, ultrametric projection reduces RMSE by >30% compared to independent per-pair HMM.

### Milestone 5: EP Loop Integration (Week 5–6)

**Goal:** Full Tier 3 pipeline with EP iterations, convergence, and posterior extraction.

Tasks:
1. Implement `convergence.cu`: message update and convergence check
2. Implement `pipeline.cu`: full streaming orchestration with pair tiles × site blocks
3. Implement `summary.cu`: posterior mean and credible interval extraction  
4. Implement `estimator.py`: full Python `CoalescenceEstimator` class
5. End-to-end test on msprime simulation

**Validation:** Tier 3 achieves correlation >0.85 with true TMRCAs on simulated data (n=200, S=10M). Convergence in ≤5 iterations.

### Milestone 6: Accuracy Benchmarking (Week 6–7)

**Goal:** Demonstrate competitive accuracy with MSMC2 and tsdate.

Tasks:
1. Run MSMC2 on same simulated data, compare pairwise TMRCAs
2. Run tsinfer + tsdate on same data, compare TMRCAs
3. Test across demographic scenarios: constant size, bottleneck, expansion, structure
4. Test with variable mutation rate (using realistic mutability maps)
5. Write benchmark report

**Target:** Within 20% of MSMC2 RMSE, while being 100-1000× faster.

### Milestone 7: Scaling and I/O (Week 7–8)

**Goal:** Multi-GPU support, VCF/PLINK input, biobank-scale demo.

Tasks:
1. Multi-GPU haplotype sharding
2. VCF streaming reader with on-the-fly bitpacking
3. PLINK bed reader
4. Test at n=10K, n=100K (simulated)
5. Test on real data: 1000 Genomes Phase 3, Ag1000G

### Milestone 8: Release (Week 8–10)

Tasks:
1. Documentation and tutorials
2. PyPI packaging
3. Benchmarks on real biobank-scale data (if accessible)
4. Preprint

---

## 8. Key Design Decisions and Alternatives

### 8.1 Why K=32 Time Bins?

- 32 threads = 1 warp. The entire HMM transition step (matrix-vector product) happens within a single warp using `__shfl_xor_sync` for the reduction. No shared memory needed for the hot loop.
- 32 bins with quadratic spacing gives resolution of ~10 generations at recent times and ~1000 generations at ancient times. This is sufficient for most applications.
- Alternative: K=64 (2 warps cooperating via shared memory). Use if deep-time resolution is critical.

### 8.2 Why Expectation Propagation and Not Variational EM?

- EP naturally decomposes into the two parallel factors (along-genome HMM, across-pair tree).
- Variational EM would require specifying and optimizing a global variational family, which is harder to parallelize.
- EP's message-passing structure maps cleanly to the GPU's streaming execution model: messages are small tensors that can be updated in-place.
- Damping ($\alpha = 0.5$) prevents oscillation, which is the main risk with EP.

### 8.3 Why Not Use CuPy/Numba/JAX Instead of Raw CUDA?

- The HMM kernel is the bottleneck and it requires warp-level intrinsics (`__shfl_xor_sync`, `__popcll`), register-resident state, and careful shared memory management that high-level frameworks can't express.
- The outer orchestration (pair tiling, site blocking, streaming) can be managed from Python, with only the inner kernels in CUDA.
- CuPy is used for ancillary operations (SFS computation, summary statistics) where performance is less critical.

### 8.4 Half-Precision (FP16) for Posteriors

- Gamma arrays dominate memory. FP16 halves the footprint, doubling the number of pairs per tile.
- Posterior probabilities are in $[0, 1]$ and sum to 1 across K bins. FP16 has sufficient precision for this (minimum subnormal is ~6e-8, far smaller than any meaningful posterior mass).
- Forward/backward computation uses FP32 in registers; only the stored gamma is FP16.
- Alternative: BF16 on Ampere+ for wider dynamic range, though FP16 is sufficient here.

### 8.5 Handling Missing Data / Variable Ploidy

- Missing data: treat as uninformative emission (emission probability = 1 for all time bins). The HMM naturally handles this — missing sites just don't update alpha.
- Diploid unphased data: for unphased genotypes (0/1/2), the emission model changes. For genotype $g \in \{0, 1, 2\}$ and two unphased diploid individuals with genotypes $g_A, g_B$, the emission depends on the joint coalescence times of all 4 haplotypes. This requires a more complex HMM state space and is out of scope for v1. Require phased input for v1.

---

## 9. Theoretical Analysis

### 9.1 Statistical Efficiency

**Within-segment MLE (Tier 2):** For a segment of physical length $L$ bp with true TMRCA $T$ and mutation rate $\mu$, the number of mutations is Poisson with mean $\lambda = 2\mu T L$. The MLE $\hat{T} = C / (2\mu L)$ has variance $\text{Var}(\hat{T}) = T / (2\mu L)$. This achieves the Cramér-Rao lower bound — no estimator can do better from mutation data alone within a segment.

**HMM posterior (Tier 3):** The HMM does better than the within-segment MLE because it borrows information from the demographic prior and from neighboring segments via the transition model. In regions with very few mutations, the prior pulls the estimate toward the population-average TMRCA, reducing variance at the cost of some bias. The posterior width correctly reflects this bias-variance tradeoff.

**Ultrametric denoising:** For a subsample of $m$ haplotypes, the $\binom{m}{2}$ pairwise TMRCAs are determined by $m - 1$ internal node times. This is $\binom{m}{2} / (m-1) \approx m/2$ observations per parameter. The denoising effect is approximately a factor of $\sqrt{m/2}$ reduction in standard deviation for shared node times (node times that affect many pairs).

### 9.2 Computational Complexity

| Tier | Time Complexity | Space Complexity | Wall-clock (n=500K, S=10M, 8×A100) |
|------|----------------|-----------------|-------------------------------------|
| 1    | $O(n^2 S / 64)$ | $O(nS/64 + S)$ | ~10 minutes |
| 2    | $O(n^2 S)$ | $O(nS/64 + n^2)$ | ~2 hours |
| 3 (subsample m=200) | $O(m^2 S K \cdot I + S m^3 \cdot I)$ | $O(m^2 S K)$ | ~5 minutes |
| 3 (all pairs) | $O(n^2 S K \cdot I)$ | $O(T_{\text{tile}} S K)$ | ~50 hours |

Where $I$ = number of EP iterations (typically 3–5), $K$ = 32 time bins, $T_{\text{tile}}$ = pair tile size.

The practical sweet spot is Tier 1 for all pairs (genome-wide $\pi$ and rough TMRCAs) plus Tier 3 for a subsample of $m = 200$–$1000$ haplotypes (accurate TMRCAs with uncertainty). This gives you both breadth and depth in minutes.

---

## 10. Connection to Existing Work and Novelty

### 10.1 Relationship to Existing Methods

| Method | What it does | Accuracy | Scale | Key limitation |
|--------|-------------|----------|-------|---------------|
| PSMC | Single-pair TMRCA via HMM | High | 1 pair | Cannot use multi-sample info |
| MSMC2 | Multi-pair TMRCA via HMM | Highest | 2–8 haplotypes | Exponential in # haplotypes |
| SMC++ | $N_e(t)$ from SFS + LD | Medium | ~200 samples | No per-site TMRCAs |
| tsinfer + tsdate | Full ARG → node dating | High | ~10K samples | Tree inference is slow, error propagates |
| ARGweaver | Full ARG via MCMC | Highest | ~50 samples | MCMC, extremely slow |
| Relate | Approx. ARG per pair | Medium-High | ~10K samples | Sequential, approximate |
| **gamma_smc_cu** | **Pairwise TMRCA via GPU variational SMC** | **High** | **Biobank (500K+)** | **Pairwise only (no full ARG)** |

### 10.2 What Is Novel

1. **GPU-native SMC inference:** No existing method runs the sequentially Markov coalescent on GPU. The batched HMM kernel with warp-level transition computation is new.

2. **Across-pair tree consistency as EP messages:** The idea of coupling independent per-pair HMMs through an ultrametric consistency constraint, via expectation propagation, is new. MSMC2 achieves cross-pair consistency by jointly modeling all pairs in a single HMM with exponentially many states. We achieve it through iterative message passing between simple per-pair HMMs, which is embarrassingly parallel.

3. **Coalescence time field:** No existing tool produces a dense $\text{pairs} \times \text{sites}$ TMRCA matrix with proper uncertainty at biobank scale. This is a new data product.

4. **Multi-resolution design:** The three-tier architecture (instant/segmented/posterior) with shared infrastructure is a new design pattern for population genetics methods.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| EP doesn't converge | Tier 3 accuracy degrades | Damping, momentum, fallback to Tier 2 |
| FP16 precision loss | Numerical instability in HMM | Log-space forward-backward, FP32 accumulation |
| Memory exceeds GPU capacity | Can't process large tiles | Reduce PAIR_TILE_SIZE or SITE_BLOCK_SIZE, stream more |
| Changepoint detection misses breakpoints | Tier 2 segments are too long, averaging over true breakpoints | Reduce BIC penalty, or skip Tier 2 and go directly to Tier 3 HMM |
| Demographic model misspecification | Prior biases TMRCA estimates | Allow flat prior mode (uniform $q_k$), or fit flexible nonparametric $N_e(t)$ |
| Ultrametric constraint too rigid (admixture, population structure) | Tree assumption violated | Relax to partial ultrametric (allow reticulation), or condition on population labels |
| CUDA portability | Only runs on NVIDIA GPUs | Future: HIP port for AMD, or SYCL for Intel |

---

## Appendix A: Mathematical Derivations

### A.1 SMC Transition Matrix Derivation

Consider two loci separated by recombination distance $r$ (in Morgans). Under the SMC, the TMRCA at the second locus depends on the TMRCA at the first locus as follows:

Let $T_1 = t$ be the TMRCA at locus 1. The probability of a recombination event in the ancestral lineage between the two loci is $1 - e^{-rt}$ (recombination can only occur in the interval $[0, t]$ where both lineages exist).

If no recombination occurs (probability $e^{-rt}$): $T_2 = T_1 = t$ (same tree).

If recombination occurs (probability $1 - e^{-rt}$): one lineage is "broken" and must find a new coalescent partner. The new coalescence time is drawn from the standard coalescent distribution (the prior $q_k$), independent of $T_1$.

Therefore:
$$P(T_2 = t_l \mid T_1 = t_k) = e^{-r t_k} \cdot \delta_{kl} + (1 - e^{-r t_k}) \cdot q_l$$

### A.2 Coalescent Prior Under Piecewise-Constant $N_e$

For piecewise-constant population size with $N_e(t) = N_m$ for $t \in [\tau_m, \tau_{m+1})$:

$$q_k = \int_{t_k}^{t_{k+1}} \lambda(t) \exp\left(-\int_0^t \lambda(u) du\right) dt$$

where $\lambda(t) = 1/N_e(t)$ is the coalescent rate.

The integral $\Lambda(t) = \int_0^t \lambda(u) du$ is piecewise linear:
$$\Lambda(t) = \sum_{m: \tau_m < t} \frac{\min(t, \tau_{m+1}) - \tau_m}{N_m}$$

Therefore:
$$q_k = \int_{t_k}^{t_{k+1}} \frac{1}{N_e(t)} e^{-\Lambda(t)} dt$$

This integral can be computed exactly as a sum of exponential terms over the $N_e$ epochs that overlap with time bin $[t_k, t_{k+1})$.

### A.3 Emission Probability Incorporating Mutation Rate Variation

For a segregating site $s$ with local mutation rate $\mu(s)$, the emission probability under the infinite-sites model:

$$P(d_s = 1 \mid T = t) = 1 - e^{-2\mu(s) t}$$

For the "gap" between consecutive segregating sites at positions $x_s$ and $x_{s+1}$, the probability of observing no mutations in the $(x_{s+1} - x_s - 1)$ intervening base pairs:

$$P(\text{gap} \mid T = t) = \prod_{b=x_s+1}^{x_{s+1}-1} e^{-2\mu(b) t} = \exp\left(-2t \sum_{b=x_s+1}^{x_{s+1}-1} \mu(b)\right)$$

If using a uniform mutation rate $\mu$, this simplifies to $\exp(-2\mu t (x_{s+1} - x_s - 1))$.

If using a per-base mutation rate map, precompute the cumulative mutation rate $M(x) = \sum_{b=0}^{x-1} \mu(b)$ (analogous to the cumulative recombination map). Then the gap emission is $\exp(-2t (M(x_{s+1}) - M(x_s) - \mu(x_s)))$.

---

## Appendix B: Testing Strategy

### B.1 Overview and Philosophy

Every CUDA kernel has a pure-numpy reference implementation in `tests/reference/`. Tests work by running both implementations on identical inputs and comparing outputs. This catches both correctness bugs and numerical drift from FP16/FP32 mixed precision.

Testing is organized into six categories, each serving a distinct purpose:

| Category | Purpose | When to run | Typical runtime |
|----------|---------|-------------|-----------------|
| Unit | Individual kernel correctness | Every commit | < 2 min |
| Integration | Multi-kernel pipelines, data flow | Every PR | < 10 min |
| Statistical | Accuracy against ground truth | Nightly / pre-release | < 30 min |
| Property | Mathematical invariants | Every commit | < 1 min |
| Regression | Catch output drift | Every commit | < 1 min |
| Performance | Speed and memory bounds | Weekly / pre-release | < 20 min |

**Test runner:** pytest with markers for category and GPU requirements.

```bash
# Fast: unit + property + regression (no GPU required for reference tests)
pytest tests/unit tests/property tests/regression -v

# Full: everything except performance
pytest tests/ -v --ignore=tests/performance

# GPU-only tests
pytest tests/ -v -m "gpu"

# Performance benchmarks (not part of CI gate, but tracked)
pytest tests/performance -v --benchmark-save=baseline

# Statistical tests (longer, use larger simulations)
pytest tests/statistical -v --runslow
```

**CI configuration:**

```yaml
# .github/workflows/test.yml
name: tests
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: ubuntu-latest  # CPU-only runner
    steps:
      - run: pytest tests/unit tests/property tests/regression -v -m "not gpu"
  
  gpu-tests:
    runs-on: [self-hosted, gpu]  # GPU runner
    steps:
      - run: pytest tests/unit tests/integration -v
  
  nightly-statistical:
    runs-on: [self-hosted, gpu]
    schedule:
      - cron: '0 3 * * *'
    steps:
      - run: pytest tests/statistical -v --runslow
```

### B.2 Shared Test Fixtures (`conftest.py`)

```python
import pytest
import numpy as np
import msprime

@pytest.fixture(scope="session")
def small_simulation():
    """
    Small tree sequence for fast unit tests.
    n=20 haplotypes, 100 kb, constant Ne=10000.
    """
    ts = msprime.sim_ancestry(
        samples=10,  # 10 diploid = 20 haploid
        sequence_length=100_000,
        recombination_rate=1e-8,
        population_size=10_000,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
    G = ts.genotype_matrix().T  # (n_haplotypes, n_sites), uint8
    positions = np.array([v.position for v in ts.variants()])
    return ts, G, positions

@pytest.fixture(scope="session")
def medium_simulation():
    """
    Medium tree sequence for integration tests.
    n=200 haplotypes, 1 Mb, bottleneck demography.
    """
    demography = msprime.Demography()
    demography.add_population(initial_size=10_000)
    demography.add_population_parameters_change(time=5000, initial_size=1000)
    demography.add_population_parameters_change(time=6000, initial_size=50_000)
    
    ts = msprime.sim_ancestry(
        samples=100,
        sequence_length=1_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=44,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=45)
    G = ts.genotype_matrix().T
    positions = np.array([v.position for v in ts.variants()])
    return ts, G, positions

@pytest.fixture(scope="session")
def large_simulation():
    """
    Large tree sequence for statistical tests. 
    n=500 haplotypes, 10 Mb, complex demography.
    Marked slow — only runs with --runslow.
    """
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=20_000)
    demography.add_population(name="B", initial_size=15_000)
    demography.add_population_parameters_change(time=2000, population="A", initial_size=5_000)
    demography.add_mass_migration(time=10_000, source="B", dest="A", proportion=1.0)
    demography.add_population_parameters_change(time=10_000, initial_size=30_000)
    
    ts = msprime.sim_ancestry(
        samples={"A": 150, "B": 100},
        sequence_length=10_000_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=46,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=47)
    G = ts.genotype_matrix().T
    positions = np.array([v.position for v in ts.variants()])
    return ts, G, positions

@pytest.fixture
def true_pairwise_tmrca(small_simulation):
    """
    Extract true pairwise TMRCA from tree sequence for a set of test pairs.
    Returns dict: (i,j) -> np.ndarray of shape (sequence_length,)
    """
    ts = small_simulation[0]
    pairs = [(0, 1), (0, 5), (1, 2), (0, 19)]
    result = {}
    for i, j in pairs:
        tmrca = np.zeros(int(ts.sequence_length))
        for tree in ts.trees():
            left, right = int(tree.interval.left), int(tree.interval.right)
            tmrca[left:right] = tree.tmrca(i, j)
        result[(i, j)] = tmrca
    return result

@pytest.fixture
def uniform_mu():
    return 1.25e-8

@pytest.fixture
def uniform_rho():
    return 1e-8
```

### B.3 Unit Tests

#### B.3.1 Bitpacking (`test_bitpack.py`)

```python
import numpy as np
import pytest

class TestBitpacking:
    """Verify bitpacking preserves genotype information exactly."""

    def test_roundtrip_small(self, small_simulation):
        """Pack and unpack, verify identity."""
        _, G, _ = small_simulation
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, G.shape[0], G.shape[1])
        np.testing.assert_array_equal(G, unpacked)

    def test_non_multiple_of_64(self):
        """Sites count not divisible by 64 — verify padding is zero-filled."""
        G = np.random.randint(0, 2, size=(10, 100), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 10, 100)
        np.testing.assert_array_equal(G, unpacked)

    def test_all_zeros(self):
        """Monomorphic zero matrix."""
        G = np.zeros((50, 1000), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 50, 1000)
        np.testing.assert_array_equal(G, unpacked)

    def test_all_ones(self):
        """Monomorphic one matrix."""
        G = np.ones((50, 1000), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 50, 1000)
        np.testing.assert_array_equal(G, unpacked)

    def test_single_site(self):
        """Edge case: S=1."""
        G = np.array([[0], [1], [1], [0]], dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 4, 1)
        np.testing.assert_array_equal(G, unpacked)

    def test_single_haplotype(self):
        """Edge case: n=1."""
        G = np.random.randint(0, 2, size=(1, 500), dtype=np.uint8)
        packed = gamma_smc_cu.bitpack(G)
        unpacked = gamma_smc_cu.unpack(packed, 1, 500)
        np.testing.assert_array_equal(G, unpacked)
```

#### B.3.2 Prefix Scan (`test_prefix_scan.py`)

```python
class TestPrefixScan:
    """Verify pairwise XOR prefix scan against numpy reference."""

    def test_prefix_scan_correctness(self, small_simulation):
        """Compare CUDA prefix scan to numpy cumsum of XOR."""
        _, G, _ = small_simulation
        pairs = [(0, 1), (2, 3), (0, 10)]
        
        for i, j in pairs:
            xor = np.bitwise_xor(G[i], G[j]).astype(np.int64)
            expected_prefix = np.cumsum(xor)
            
            cuda_prefix = gamma_smc_cu.pairwise_prefix_scan(G, [(i, j)])[0]
            np.testing.assert_array_equal(cuda_prefix, expected_prefix)

    def test_windowed_divergence_from_prefix(self, small_simulation):
        """Verify windowed divergence = prefix[s+W] - prefix[s-W]."""
        _, G, positions = small_simulation
        W = 50  # window of 50 sites
        
        xor = np.bitwise_xor(G[0], G[1]).astype(np.int64)
        prefix = np.cumsum(xor)
        
        for s in range(W, len(positions) - W):
            expected = prefix[s + W] - prefix[s - W]
            cuda_div = gamma_smc_cu.windowed_divergence(G, [(0, 1)], W)[0, s]
            assert cuda_div == expected

    def test_symmetry(self, small_simulation):
        """prefix_scan(i,j) == prefix_scan(j,i) since XOR is symmetric."""
        _, G, _ = small_simulation
        p_ij = gamma_smc_cu.pairwise_prefix_scan(G, [(0, 5)])[0]
        p_ji = gamma_smc_cu.pairwise_prefix_scan(G, [(5, 0)])[0]
        np.testing.assert_array_equal(p_ij, p_ji)

    def test_self_pair_is_zero(self, small_simulation):
        """XOR of haplotype with itself is zero everywhere."""
        _, G, _ = small_simulation
        prefix = gamma_smc_cu.pairwise_prefix_scan(G, [(3, 3)])[0]
        np.testing.assert_array_equal(prefix, np.zeros_like(prefix))
```

#### B.3.3 SFS (`test_sfs.py`)

```python
class TestSFS:
    """Verify SFS computation against tskit."""

    def test_sfs_matches_tskit(self, small_simulation):
        ts, G, _ = small_simulation
        cuda_sfs = gamma_smc_cu.compute_sfs(G)
        tskit_sfs = ts.allele_frequency_spectrum(polarised=True, span_normalise=False)
        np.testing.assert_array_equal(cuda_sfs, tskit_sfs.astype(int))

    def test_sfs_sums_to_num_sites(self, small_simulation):
        _, G, _ = small_simulation
        sfs = gamma_smc_cu.compute_sfs(G)
        # SFS entries 1..n-1 should sum to total segregating sites
        assert sfs[1:-1].sum() == G.shape[1]

    def test_sfs_monomorphic_input(self):
        """All-zero matrix should give empty SFS."""
        G = np.zeros((20, 100), dtype=np.uint8)
        sfs = gamma_smc_cu.compute_sfs(G)
        assert sfs[1:].sum() == 0
```

#### B.3.4 HMM Forward-Backward (`test_hmm.py`)

```python
from tests.reference.hmm_numpy import NumpyHMM

class TestHMMForwardBackward:
    """
    Verify CUDA HMM against pure-numpy reference implementation.
    This is the most critical test suite in the project.
    """

    @pytest.fixture
    def numpy_hmm(self, small_simulation, uniform_mu, uniform_rho):
        ts, G, positions = small_simulation
        return NumpyHMM(G, positions, mu=uniform_mu, rho=uniform_rho, K=32, Ne=10_000)

    def test_forward_probabilities(self, small_simulation, numpy_hmm):
        """Forward probabilities match numpy reference."""
        _, G, positions = small_simulation
        pair = (0, 1)
        
        np_alpha = numpy_hmm.forward(pair)  # (S, K)
        cuda_alpha = gamma_smc_cu.hmm_forward(G, positions, pair, K=32)
        
        np.testing.assert_allclose(cuda_alpha, np_alpha, rtol=1e-4, atol=1e-6)

    def test_backward_probabilities(self, small_simulation, numpy_hmm):
        """Backward probabilities match numpy reference."""
        _, G, positions = small_simulation
        pair = (0, 1)
        
        np_beta = numpy_hmm.backward(pair)
        cuda_beta = gamma_smc_cu.hmm_backward(G, positions, pair, K=32)
        
        np.testing.assert_allclose(cuda_beta, np_beta, rtol=1e-4, atol=1e-6)

    def test_posterior_marginals(self, small_simulation, numpy_hmm):
        """Posterior gamma = alpha * beta, normalized, matches reference."""
        _, G, positions = small_simulation
        pair = (0, 1)
        
        np_gamma = numpy_hmm.posterior(pair)  # (S, K)
        cuda_gamma = gamma_smc_cu.hmm_posterior(G, positions, pair, K=32)
        
        np.testing.assert_allclose(cuda_gamma, np_gamma, rtol=1e-3, atol=1e-5)

    def test_posterior_sums_to_one(self, small_simulation):
        """Posterior marginals must sum to 1 at every site."""
        _, G, positions = small_simulation
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32)
        sums = gamma.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-4)

    def test_log_likelihood(self, small_simulation, numpy_hmm):
        """Total log-likelihood matches reference."""
        _, G, positions = small_simulation
        pair = (0, 1)
        
        np_ll = numpy_hmm.log_likelihood(pair)
        cuda_ll = gamma_smc_cu.hmm_log_likelihood(G, positions, pair, K=32)
        
        np.testing.assert_allclose(cuda_ll, np_ll, rtol=1e-3)

    def test_batched_matches_single(self, small_simulation):
        """Batched HMM (multiple pairs) matches running pairs individually."""
        _, G, positions = small_simulation
        pairs = [(0, 1), (2, 3), (0, 10), (5, 15)]
        
        # Run batched
        batched_gamma = gamma_smc_cu.hmm_posterior_batched(G, positions, pairs, K=32)
        
        # Run individually
        for idx, pair in enumerate(pairs):
            single_gamma = gamma_smc_cu.hmm_posterior(G, positions, pair, K=32)
            np.testing.assert_allclose(
                batched_gamma[idx], single_gamma, rtol=1e-4,
                err_msg=f"Mismatch for pair {pair}"
            )

    def test_uniform_prior_gives_flat_posterior_no_data(self):
        """With no mutations (all-zero genotypes), posterior should equal prior."""
        n, S = 10, 1000
        G = np.zeros((n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100
        
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32)
        # With no data, posterior should be close to prior at all sites
        # (transition model spreads prior, but shouldn't deviate far)
        prior = gamma_smc_cu.coalescent_prior(Ne=10_000, K=32)
        for s in range(S):
            np.testing.assert_allclose(gamma[s], prior, atol=0.05)

    def test_high_divergence_pair_has_large_tmrca(self, small_simulation):
        """Pairs with many differences should have larger posterior mean TMRCA."""
        _, G, positions = small_simulation
        n = G.shape[0]
        
        # Find pair with most and fewest differences
        max_diff, min_diff = 0, G.shape[1]
        max_pair, min_pair = (0, 1), (0, 1)
        for i in range(min(20, n)):
            for j in range(i+1, min(20, n)):
                d = np.sum(G[i] != G[j])
                if d > max_diff:
                    max_diff, max_pair = d, (i, j)
                if d < min_diff:
                    min_diff, min_pair = d, (i, j)
        
        gamma_max = gamma_smc_cu.hmm_posterior(G, positions, max_pair, K=32)
        gamma_min = gamma_smc_cu.hmm_posterior(G, positions, min_pair, K=32)
        
        t_mid = gamma_smc_cu.time_midpoints(K=32)
        mean_max = np.mean(gamma_max @ t_mid)
        mean_min = np.mean(gamma_min @ t_mid)
        
        assert mean_max > mean_min, \
            f"High-divergence pair TMRCA ({mean_max:.0f}) should exceed low-divergence ({mean_min:.0f})"
```

#### B.3.5 HMM Numerical Stability (`test_hmm_numerical.py`)

```python
class TestHMMNumericalStability:
    """
    Edge cases that stress numerical precision: very long sequences, 
    extreme TMRCA values, near-zero emissions.
    """

    def test_long_monomorphic_stretch(self):
        """
        10 Mb of no mutations. Forward probabilities must not underflow.
        Verifies log-space rescaling is working.
        """
        n, S = 4, 10
        G = np.zeros((n, S), dtype=np.uint8)
        G[0, 0] = 1  # single mutation at start
        G[0, -1] = 1  # single mutation at end
        positions = np.linspace(0, 10_000_000, S)
        
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32)
        
        # Must not contain NaN or Inf
        assert np.all(np.isfinite(gamma)), "NaN/Inf in posterior after long monomorphic stretch"
        # Must still sum to 1
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-3)

    def test_dense_mutations(self):
        """
        Every site is a mutation (saturated divergence).
        Posterior should concentrate on large time bins.
        """
        n, S = 4, 5000
        G = np.zeros((n, S), dtype=np.uint8)
        G[0, :] = 1  # haplotype 0 differs from all others at every site
        positions = np.arange(S, dtype=np.float64)
        
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32)
        assert np.all(np.isfinite(gamma))
        
        t_mid = gamma_smc_cu.time_midpoints(K=32)
        mean_tmrca = np.mean(gamma @ t_mid)
        assert mean_tmrca > t_mid[K // 2], "Saturated divergence should give large TMRCA"

    def test_fp16_vs_fp32_gamma(self, small_simulation):
        """
        Verify FP16 stored gamma doesn't deviate too far from FP32 computation.
        """
        _, G, positions = small_simulation
        gamma_fp32 = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32, precision="fp32")
        gamma_fp16 = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32, precision="fp16")
        
        # FP16 has ~3 decimal digits of precision
        np.testing.assert_allclose(gamma_fp16, gamma_fp32, rtol=5e-3, atol=1e-4)

    def test_very_small_ne(self):
        """Ne = 100 — very recent coalescence, tests small time bins."""
        n, S = 10, 1000
        G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100
        
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32, Ne=100)
        assert np.all(np.isfinite(gamma))

    def test_very_large_ne(self):
        """Ne = 1e7 — very ancient coalescence, tests large time bins."""
        n, S = 10, 1000
        G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100
        
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32, Ne=10_000_000)
        assert np.all(np.isfinite(gamma))
```

#### B.3.6 Ultrametric Projection (`test_ultrametric.py`)

```python
from tests.reference.ultrametric_numpy import NumpyUltrametric

class TestUltrametricProjection:
    """Verify ultrametric tree fitting against numpy reference."""

    def test_already_ultrametric_input(self):
        """If input posteriors are delta functions on a valid tree, output is unchanged."""
        K = 32
        m = 6
        t_mid = gamma_smc_cu.time_midpoints(K=K)
        
        # Construct a known tree: ((0,1):t5, (2,3):t10, ((0,1),(2,3)):t20, (4,5):t15)
        # True pairwise TMRCAs:
        true_t = {
            (0,1): 5, (2,3): 10, (4,5): 15,
            (0,2): 20, (0,3): 20, (1,2): 20, (1,3): 20,
            (0,4): 25, (0,5): 25, (1,4): 25, (1,5): 25,
            (2,4): 25, (2,5): 25, (3,4): 25, (3,5): 25,
        }
        
        # Create delta-function posteriors at the true time bins
        posteriors = {}
        for pair, t in true_t.items():
            k = np.argmin(np.abs(t_mid - t))
            post = np.zeros(K)
            post[k] = 1.0
            posteriors[pair] = post
        
        result = gamma_smc_cu.ultrametric_project(posteriors, m=m, K=K)
        
        for pair in true_t:
            k_true = np.argmin(np.abs(t_mid - true_t[pair]))
            k_result = np.argmax(result[pair])
            assert k_result == k_true, f"Pair {pair}: expected bin {k_true}, got {k_result}"

    def test_denoising_effect(self):
        """Noisy posteriors should be improved by ultrametric constraint."""
        K = 32
        m = 10
        
        # Create noisy posteriors from a known tree
        np_ref = NumpyUltrametric(K=K)
        true_tree, true_posteriors = np_ref.generate_noisy_tree(m=m, noise_level=0.3)
        
        projected = gamma_smc_cu.ultrametric_project(true_posteriors, m=m, K=K)
        
        # Projected should be closer to truth than raw posteriors
        error_raw = np_ref.ultrametric_violation(true_posteriors)
        error_proj = np_ref.ultrametric_violation(projected)
        
        assert error_proj < error_raw, \
            f"Projection should reduce ultrametric violation: {error_proj:.4f} >= {error_raw:.4f}"

    def test_output_matches_numpy_reference(self):
        """CUDA kernel output matches numpy agglomerative clustering."""
        K = 32
        m = 8
        np_ref = NumpyUltrametric(K=K)
        _, posteriors = np_ref.generate_noisy_tree(m=m, noise_level=0.2, seed=42)
        
        cuda_result = gamma_smc_cu.ultrametric_project(posteriors, m=m, K=K)
        numpy_result = np_ref.project(posteriors, m=m)
        
        for pair in posteriors:
            np.testing.assert_allclose(
                cuda_result[pair], numpy_result[pair], atol=0.05,
                err_msg=f"Mismatch for pair {pair}"
            )
```

#### B.3.7 PELT Changepoint Detection (`test_pelt.py`)

```python
from tests.reference.pelt_numpy import NumpyPELT

class TestPELT:
    """Verify PELT changepoint detection against numpy reference."""

    def test_known_changepoints(self):
        """Synthetic signal with known breakpoints."""
        # Create piecewise-constant rate signal
        rates = [0.001, 0.01, 0.001, 0.05, 0.001]
        segment_lengths = [10000, 5000, 20000, 3000, 12000]
        
        signal = np.zeros(sum(segment_lengths), dtype=np.uint8)
        pos = 0
        true_breakpoints = []
        for rate, length in zip(rates, segment_lengths):
            mutations = np.random.binomial(1, rate, size=length)
            signal[pos:pos+length] = mutations
            pos += length
            true_breakpoints.append(pos)
        
        detected = gamma_smc_cu.pelt_changepoints(signal)
        
        # Each detected breakpoint should be within 500 bp of a true one
        for bp in detected:
            distances = [abs(bp - tbp) for tbp in true_breakpoints[:-1]]
            assert min(distances) < 500, f"Detected breakpoint {bp} far from any true breakpoint"

    def test_no_changepoints(self):
        """Constant rate signal should produce no changepoints (or very few)."""
        signal = np.random.binomial(1, 0.005, size=50000).astype(np.uint8)
        detected = gamma_smc_cu.pelt_changepoints(signal)
        assert len(detected) <= 2, f"Expected ≤2 spurious changepoints, got {len(detected)}"

    def test_matches_numpy_reference(self, small_simulation):
        """CUDA PELT matches numpy PELT on real genotype data."""
        _, G, positions = small_simulation
        xor = np.bitwise_xor(G[0], G[1]).astype(np.uint8)
        
        np_pelt = NumpyPELT()
        np_breakpoints = np_pelt.detect(xor, positions)
        cuda_breakpoints = gamma_smc_cu.pelt_changepoints(xor, positions)
        
        np.testing.assert_array_equal(cuda_breakpoints, np_breakpoints)
```

#### B.3.8 Pair Indexing (`test_pair_indexing.py`)

```python
class TestPairIndexing:
    """Verify bijection between linear pair index and (i,j) tuple."""

    @pytest.mark.parametrize("n", [4, 10, 50, 100, 1000])
    def test_roundtrip(self, n):
        """pair_to_index(index_to_pair(p)) == p for all valid p."""
        n_pairs = n * (n - 1) // 2
        for p in range(min(n_pairs, 10000)):  # test first 10K
            i, j = gamma_smc_cu.index_to_pair(p)
            assert i > j, f"Convention: i > j, got i={i}, j={j}"
            assert gamma_smc_cu.pair_to_index(i, j) == p

    @pytest.mark.parametrize("n", [4, 10, 100])
    def test_all_pairs_covered(self, n):
        """All (i,j) pairs with i > j are reachable."""
        n_pairs = n * (n - 1) // 2
        seen = set()
        for p in range(n_pairs):
            pair = gamma_smc_cu.index_to_pair(p)
            seen.add(pair)
        
        expected = {(i, j) for i in range(n) for j in range(i)}
        assert seen == expected
```

#### B.3.9 Emission and Transition Probabilities (`test_emissions.py`, `test_transitions.py`)

```python
class TestEmissions:
    """Verify emission probability computation."""

    def test_emission_mutation_site(self):
        """P(d=1 | T=t) = 1 - exp(-2μt)."""
        mu = 1.25e-8
        t = 10_000.0
        expected = 1.0 - np.exp(-2 * mu * t)
        computed = gamma_smc_cu.emission_prob(d=1, t=t, mu=mu)
        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    def test_emission_no_mutation_site(self):
        """P(d=0 | T=t) = exp(-2μt)."""
        mu = 1.25e-8
        t = 10_000.0
        expected = np.exp(-2 * mu * t)
        computed = gamma_smc_cu.emission_prob(d=0, t=t, mu=mu)
        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    def test_emissions_sum_to_one(self):
        """P(d=0|t) + P(d=1|t) = 1 for all t."""
        mu = 1.25e-8
        for t in [10, 1000, 100_000, 1_000_000]:
            p0 = gamma_smc_cu.emission_prob(d=0, t=t, mu=mu)
            p1 = gamma_smc_cu.emission_prob(d=1, t=t, mu=mu)
            np.testing.assert_allclose(p0 + p1, 1.0, rtol=1e-12)

    def test_gap_emission(self):
        """Gap emission for L bp with no mutations."""
        mu = 1.25e-8
        t = 10_000.0
        L = 5000
        expected = np.exp(-2 * mu * t * L)
        computed = gamma_smc_cu.gap_emission(t=t, mu=mu, gap_bp=L)
        np.testing.assert_allclose(computed, expected, rtol=1e-10)


class TestTransitions:
    """Verify SMC transition matrix construction."""

    def test_transition_rows_sum_to_one(self):
        """Each row of A(r) must sum to 1."""
        K = 32
        Ne = 10_000
        coal_prior = gamma_smc_cu.coalescent_prior(Ne=Ne, K=K)
        t_mid = gamma_smc_cu.time_midpoints(K=K)
        
        for r in [1e-6, 1e-4, 1e-2, 1.0]:
            A = gamma_smc_cu.transition_matrix(r=r, t_mid=t_mid, coal_prior=coal_prior)
            row_sums = A.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10,
                err_msg=f"Row sums not 1 for r={r}")

    def test_no_recombination_is_identity(self):
        """A(r=0) should be the identity matrix."""
        K = 32
        coal_prior = gamma_smc_cu.coalescent_prior(Ne=10_000, K=K)
        t_mid = gamma_smc_cu.time_midpoints(K=K)
        A = gamma_smc_cu.transition_matrix(r=0.0, t_mid=t_mid, coal_prior=coal_prior)
        np.testing.assert_allclose(A, np.eye(K), atol=1e-10)

    def test_large_recombination_approaches_prior(self):
        """A(r→∞) should have all rows equal to coal_prior."""
        K = 32
        coal_prior = gamma_smc_cu.coalescent_prior(Ne=10_000, K=K)
        t_mid = gamma_smc_cu.time_midpoints(K=K)
        A = gamma_smc_cu.transition_matrix(r=100.0, t_mid=t_mid, coal_prior=coal_prior)
        
        for k in range(K):
            np.testing.assert_allclose(A[k], coal_prior, atol=1e-4,
                err_msg=f"Row {k} doesn't approach prior for large r")

    def test_coalescent_prior_sums_to_one(self):
        """Coalescent prior q[k] must sum to 1."""
        for Ne in [100, 10_000, 1_000_000]:
            q = gamma_smc_cu.coalescent_prior(Ne=Ne, K=32)
            np.testing.assert_allclose(q.sum(), 1.0, rtol=1e-8)
```

### B.4 Integration Tests

#### B.4.1 End-to-End Tier Pipelines

```python
class TestTier1Pipeline:
    """End-to-end Tier 1: genotype matrix → per-site π."""

    def test_pi_matches_tskit(self, small_simulation, uniform_mu):
        ts, G, positions = small_simulation
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu)
        pi = fc.site_pi()
        
        # Compare against tskit diversity
        tskit_pi = ts.diversity(mode="site", span_normalise=False)
        
        # Not exact because tskit uses a different windowing,
        # but genome-wide average should be close
        np.testing.assert_allclose(pi.mean(), tskit_pi.mean(), rtol=0.05)

    def test_tier1_output_shape(self, small_simulation, uniform_mu):
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu)
        pi = fc.site_pi()
        assert pi.shape == (len(positions),)
        assert np.all(pi >= 0)


class TestTier2Pipeline:
    """End-to-end Tier 2: genotype matrix → segments."""

    def test_segments_cover_genome(self, small_simulation, uniform_mu, uniform_rho):
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        segments = fc.segment_tmrca(pairs=[(0, 1)])
        
        segs = segments[(0, 1)]
        # Segments should tile the genome without gaps
        for i in range(len(segs) - 1):
            assert segs[i].end == segs[i + 1].start, \
                f"Gap between segment {i} and {i+1}"
        assert segs[0].start == 0
        assert segs[-1].end == len(positions) - 1

    def test_segment_tmrca_positive(self, small_simulation, uniform_mu, uniform_rho):
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        segments = fc.segment_tmrca(pairs=[(0, 1)])
        
        for seg in segments[(0, 1)]:
            assert seg.tmrca > 0, f"Segment TMRCA must be positive, got {seg.tmrca}"


class TestTier3Pipeline:
    """End-to-end Tier 3: genotype matrix → posterior TMRCAs."""

    def test_posterior_output_structure(self, small_simulation, uniform_mu, uniform_rho):
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        result = fc.infer_tmrca(pairs=[(0, 1), (2, 3)], max_iterations=2)
        
        assert result.tmrca_mean.shape == (2, len(positions))
        assert result.tmrca_lower.shape == (2, len(positions))
        assert result.tmrca_upper.shape == (2, len(positions))
        assert np.all(result.tmrca_lower <= result.tmrca_mean)
        assert np.all(result.tmrca_mean <= result.tmrca_upper)
        assert np.all(result.tmrca_mean > 0)

    def test_credible_intervals_contain_mean(self, small_simulation, uniform_mu, uniform_rho):
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=3)
        
        assert np.all(result.tmrca_lower[0] <= result.tmrca_mean[0])
        assert np.all(result.tmrca_mean[0] <= result.tmrca_upper[0])
```

#### B.4.2 EP Convergence (`test_ep_convergence.py`)

```python
class TestEPConvergence:
    """Verify expectation propagation converges and improves accuracy."""

    def test_convergence_flag(self, medium_simulation, uniform_mu, uniform_rho):
        """EP should converge within max_iterations on well-behaved data."""
        _, G, positions = medium_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        result = fc.infer_tmrca(
            pairs="subsample", subsample_size=20,
            max_iterations=10, convergence_tol=0.01,
        )
        assert result.converged, "EP did not converge within 10 iterations"

    def test_accuracy_improves_with_iterations(self, small_simulation, 
                                                true_pairwise_tmrca, 
                                                uniform_mu, uniform_rho):
        """More EP iterations should reduce error against ground truth."""
        ts, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        
        errors = []
        for n_iter in [0, 1, 3, 5]:
            result = fc.infer_tmrca(
                pairs=[(0, 1)], max_iterations=n_iter, subsample_size=10,
            )
            est = np.interp(np.arange(int(ts.sequence_length)), positions, result.tmrca_mean[0])
            true = true_pairwise_tmrca[(0, 1)]
            rmse = np.sqrt(np.mean((est - true) ** 2))
            errors.append(rmse)
        
        # Error should generally decrease (allow some non-monotonicity)
        assert errors[-1] < errors[0] * 0.9, \
            f"EP didn't improve: errors = {errors}"

    def test_damping_prevents_oscillation(self, small_simulation, uniform_mu, uniform_rho):
        """With damping=0.5, max_delta should decrease monotonically (roughly)."""
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        result = fc.infer_tmrca(
            pairs="subsample", subsample_size=10,
            max_iterations=10, damping=0.5,
            return_diagnostics=True,
        )
        deltas = result.diagnostics["max_delta_per_iteration"]
        # Allow at most one increase
        increases = sum(1 for i in range(1, len(deltas)) if deltas[i] > deltas[i-1])
        assert increases <= 2, f"Too many delta increases (oscillation): {deltas}"
```

#### B.4.3 Site Block Boundaries (`test_site_blocking.py`)

```python
class TestSiteBlocking:
    """Verify HMM produces identical results regardless of site block size."""

    def test_blocking_invariance(self, small_simulation, uniform_mu, uniform_rho):
        """
        HMM posteriors must be identical whether processed in one block
        or multiple blocks.
        """
        _, G, positions = small_simulation
        pair = (0, 1)
        
        # Full sequence in one block
        gamma_full = gamma_smc_cu.hmm_posterior(
            G, positions, pair, K=32, site_block_size=len(positions)
        )
        
        # Split into small blocks
        gamma_blocked = gamma_smc_cu.hmm_posterior(
            G, positions, pair, K=32, site_block_size=256
        )
        
        np.testing.assert_allclose(gamma_full, gamma_blocked, rtol=1e-3, atol=1e-5,
            err_msg="Site blocking changed HMM posteriors")
```

#### B.4.4 I/O Format Round-Trips (`test_io_formats.py`)

```python
class TestIOFormats:
    """Verify all input formats produce consistent results."""

    def test_tskit_input(self, small_simulation, uniform_mu, uniform_rho):
        ts, G, positions = small_simulation
        
        # From numpy
        fc_numpy = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        pi_numpy = fc_numpy.site_pi()
        
        # From tree sequence
        fc_ts = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts)
        pi_ts = fc_ts.site_pi()
        
        np.testing.assert_allclose(pi_numpy, pi_ts, rtol=1e-10)

    def test_vcf_roundtrip(self, small_simulation, tmp_path, uniform_mu, uniform_rho):
        """Write tree sequence to VCF, read back, verify consistency."""
        ts, _, _ = small_simulation
        vcf_path = tmp_path / "test.vcf"
        
        with open(vcf_path, "w") as f:
            ts.write_vcf(f)
        
        fc = gamma_smc_cu.CoalescenceEstimator.from_vcf(str(vcf_path), mu=uniform_mu, rho=uniform_rho)
        pi = fc.site_pi()
        
        assert pi.shape[0] == ts.num_sites
        assert np.all(np.isfinite(pi))
```

### B.5 Statistical Tests

```python
@pytest.mark.slow
class TestAccuracyMsprime:
    """
    Validate TMRCA estimation accuracy against msprime ground truth
    across demographic scenarios.
    """

    @pytest.mark.parametrize("scenario", [
        {"name": "constant", "Ne": 10_000, "samples": 50},
        {"name": "bottleneck", "changes": [(5000, 1000), (6000, 50_000)], "samples": 50},
        {"name": "expansion", "changes": [(1000, 100_000)], "samples": 50},
        {"name": "decline", "changes": [(2000, 500)], "samples": 50},
    ])
    def test_tmrca_correlation_by_scenario(self, scenario):
        """Correlation with true TMRCA should exceed 0.80 for all scenarios."""
        # Build demography
        demography = msprime.Demography()
        demography.add_population(initial_size=scenario.get("Ne", 10_000))
        for time, size in scenario.get("changes", []):
            demography.add_population_parameters_change(time=time, initial_size=size)
        
        ts = msprime.sim_ancestry(
            samples=scenario["samples"],
            sequence_length=5_000_000,
            recombination_rate=1e-8,
            demography=demography,
            random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
        
        fc = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts)
        result = fc.infer_tmrca(pairs=[(0, 1), (0, 10), (5, 15)], max_iterations=5)
        
        for idx, pair in enumerate([(0, 1), (0, 10), (5, 15)]):
            true_t = extract_true_tmrca(ts, pair)
            est_t = np.interp(
                np.arange(int(ts.sequence_length)),
                np.array([v.position for v in ts.variants()]),
                result.tmrca_mean[idx]
            )
            r = np.corrcoef(true_t, est_t)[0, 1]
            assert r > 0.80, \
                f"Scenario '{scenario['name']}', pair {pair}: r={r:.3f} < 0.80"

    def test_posterior_calibration(self, large_simulation):
        """
        95% credible intervals should contain the true TMRCA ~95% of the time.
        Tests posterior calibration (not just point estimate accuracy).
        """
        ts, G, positions = large_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        
        test_pairs = [(0, 1), (0, 50), (10, 100)]
        result = fc.infer_tmrca(pairs=test_pairs, max_iterations=5, subsample_size=50)
        
        coverages = []
        for idx, pair in enumerate(test_pairs):
            true_t = extract_true_tmrca(ts, pair)
            est_lower = np.interp(np.arange(len(true_t)), positions, result.tmrca_lower[idx])
            est_upper = np.interp(np.arange(len(true_t)), positions, result.tmrca_upper[idx])
            
            covered = np.mean((true_t >= est_lower) & (true_t <= est_upper))
            coverages.append(covered)
        
        mean_coverage = np.mean(coverages)
        # Allow some slack: 95% CI should give 80-100% coverage
        # (undercoverage is worse than overcoverage)
        assert mean_coverage > 0.75, \
            f"Poor calibration: mean coverage = {mean_coverage:.2f}, expected ~0.95"
        assert mean_coverage < 1.0, \
            f"Intervals too wide: coverage = {mean_coverage:.2f}"

    def test_ne_recovery(self):
        """
        Estimated Ne(t) from SFS should recover the simulated demography.
        """
        true_ne = [(0, 20_000), (5000, 2_000), (7000, 50_000)]
        
        demography = msprime.Demography()
        demography.add_population(initial_size=true_ne[0][1])
        for time, size in true_ne[1:]:
            demography.add_population_parameters_change(time=time, initial_size=size)
        
        ts = msprime.sim_ancestry(
            samples=200, sequence_length=50_000_000,
            recombination_rate=1e-8, demography=demography, random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
        
        fc = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts)
        estimated_ne = fc.estimate_demography()
        
        # Check Ne at key timepoints
        for time, true_size in true_ne:
            est_size = estimated_ne.at(time)
            ratio = est_size / true_size
            assert 0.3 < ratio < 3.0, \
                f"Ne at t={time}: true={true_size}, est={est_size:.0f}, ratio={ratio:.2f}"
```

### B.6 Property-Based Tests

```python
class TestInvariants:
    """
    Mathematical invariants that must hold regardless of input.
    These catch subtle bugs that specific test cases might miss.
    """

    def test_exchangeability(self, small_simulation, uniform_mu, uniform_rho):
        """
        Permuting sample indices should not change per-site summary statistics.
        π(s) is invariant to sample permutation.
        """
        _, G, positions = small_simulation
        
        fc1 = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        pi1 = fc1.site_pi()
        
        # Permute samples
        perm = np.random.permutation(G.shape[0])
        G_perm = G[perm]
        fc2 = gamma_smc_cu.CoalescenceEstimator(G_perm, positions, mu=uniform_mu, rho=uniform_rho)
        pi2 = fc2.site_pi()
        
        np.testing.assert_allclose(pi1, pi2, rtol=1e-10)

    def test_tmrca_symmetry(self, small_simulation, uniform_mu, uniform_rho):
        """T(i,j) == T(j,i) for all pairs."""
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        
        result_ij = fc.infer_tmrca(pairs=[(0, 5)], max_iterations=2)
        result_ji = fc.infer_tmrca(pairs=[(5, 0)], max_iterations=2)
        
        np.testing.assert_allclose(result_ij.tmrca_mean, result_ji.tmrca_mean, rtol=1e-5)

    def test_posterior_normalization(self, small_simulation, uniform_mu, uniform_rho):
        """Posterior marginals sum to 1 at every site for every pair."""
        _, G, positions = small_simulation
        gamma = gamma_smc_cu.hmm_posterior(G, positions, (0, 1), K=32)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, rtol=1e-4)

    def test_tmrca_positivity(self, small_simulation, uniform_mu, uniform_rho):
        """All TMRCA estimates must be strictly positive."""
        _, G, positions = small_simulation
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=uniform_mu, rho=uniform_rho)
        result = fc.infer_tmrca(pairs=[(0, 1), (2, 3)], max_iterations=2)
        assert np.all(result.tmrca_mean > 0)
        assert np.all(result.tmrca_lower > 0)

    def test_identical_haplotypes_give_zero_divergence(self):
        """If two haplotypes are identical, windowed divergence is 0 everywhere."""
        G = np.zeros((10, 1000), dtype=np.uint8)
        G[0, :] = np.random.randint(0, 2, 1000)
        G[1, :] = G[0, :]  # identical
        positions = np.arange(1000, dtype=np.float64) * 100
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8)
        div = fc.pairwise_divergence(pairs=[(0, 1)], window_sizes=[100])
        np.testing.assert_array_equal(div, 0)

    def test_scaling_with_mu(self, small_simulation, uniform_rho):
        """Doubling μ should approximately halve estimated TMRCA (Tier 1)."""
        _, G, positions = small_simulation
        
        fc1 = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1e-8, rho=uniform_rho)
        fc2 = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=2e-8, rho=uniform_rho)
        
        div1 = fc1.pairwise_divergence(pairs=[(0, 1)], window_sizes=[500])
        div2 = fc2.pairwise_divergence(pairs=[(0, 1)], window_sizes=[500])
        
        # Raw divergence counts are the same, but TMRCA = div / (2μ)
        # so TMRCA with 2μ should be half
        tmrca1 = div1.mean() / (2 * 1e-8)
        tmrca2 = div2.mean() / (2 * 2e-8)
        np.testing.assert_allclose(tmrca2, tmrca1 / 2, rtol=0.01)


class TestEdgeCases:
    """Degenerate inputs that should not crash."""

    def test_single_site(self):
        G = np.array([[0], [1]], dtype=np.uint8)
        positions = np.array([500.0])
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8)
        pi = fc.site_pi()
        assert pi.shape == (1,)

    def test_two_haplotypes(self):
        """Minimum sample size: n=2."""
        G = np.random.randint(0, 2, size=(2, 1000), dtype=np.uint8)
        positions = np.arange(1000, dtype=np.float64) * 100
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=2)
        assert result.tmrca_mean.shape == (1, 1000)

    def test_monomorphic_matrix(self):
        """All-zero genotypes: no segregating sites in practice, but should not crash."""
        G = np.zeros((20, 500), dtype=np.uint8)
        positions = np.arange(500, dtype=np.float64) * 100
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8)
        pi = fc.site_pi()
        np.testing.assert_array_equal(pi, 0)

    def test_all_singletons(self):
        """Every mutation is a singleton — tests extreme SFS."""
        n, S = 20, 100
        G = np.zeros((n, S), dtype=np.uint8)
        for s in range(S):
            G[s % n, s] = 1  # each site is a singleton on a different haplotype
        positions = np.arange(S, dtype=np.float64) * 1000
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=2)
        assert np.all(np.isfinite(result.tmrca_mean))

    def test_very_large_positions(self):
        """Positions spanning an entire chromosome (~250 Mb)."""
        G = np.random.randint(0, 2, size=(10, 100), dtype=np.uint8)
        positions = np.sort(np.random.uniform(0, 250_000_000, size=100))
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        pi = fc.site_pi()
        assert np.all(np.isfinite(pi))
```

### B.7 Regression Tests

```python
class TestRegressionGolden:
    """
    Compare outputs against frozen 'golden' reference values.
    Catches unintentional changes to numerical behavior across refactors.
    
    Golden files are generated once from a validated run and committed
    to the repository. Regenerate with:
        pytest tests/regression --regenerate-golden
    """

    GOLDEN_DIR = Path(__file__).parent / "golden"

    @pytest.fixture(scope="class")
    def golden_data(self):
        return np.load(self.GOLDEN_DIR / "small_sim.npz")

    @pytest.fixture(scope="class")
    def golden_posteriors(self):
        return np.load(self.GOLDEN_DIR / "expected_posteriors.npz")

    def test_tier1_pi_matches_golden(self, golden_data):
        G = golden_data["G"]
        positions = golden_data["positions"]
        expected_pi = golden_data["pi"]
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8)
        pi = fc.site_pi()
        
        np.testing.assert_allclose(pi, expected_pi, rtol=1e-6,
            err_msg="Tier 1 π output changed from golden reference")

    def test_tier3_posterior_matches_golden(self, golden_data, golden_posteriors):
        G = golden_data["G"]
        positions = golden_data["positions"]
        expected_mean = golden_posteriors["tmrca_mean"]
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=5, subsample_size=10)
        
        np.testing.assert_allclose(result.tmrca_mean[0], expected_mean, rtol=1e-3,
            err_msg="Tier 3 posterior mean changed from golden reference")
```

### B.8 Performance Tests

```python
@pytest.mark.performance
class TestPerformance:
    """
    Performance regression tests. These don't gate CI but are tracked
    over time to catch performance regressions.
    """

    @pytest.mark.gpu
    def test_tier1_throughput(self, benchmark):
        """Tier 1 should process >1M sites/second for 1K haplotypes."""
        n, S = 1000, 1_000_000
        G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64)
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8)
        
        result = benchmark(fc.site_pi)
        
        sites_per_second = S / benchmark.stats["mean"]
        assert sites_per_second > 1_000_000, \
            f"Tier 1 throughput {sites_per_second:.0f} sites/s < 1M target"

    @pytest.mark.gpu
    def test_hmm_pairs_per_second(self, benchmark):
        """Batched HMM should process >100 pairs/second on 100K sites."""
        n, S = 100, 100_000
        G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100
        pairs = [(i, j) for i in range(20) for j in range(i)]  # 190 pairs
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        
        result = benchmark(lambda: fc.infer_tmrca(pairs=pairs, max_iterations=1))
        
        pairs_per_second = len(pairs) / benchmark.stats["mean"]
        assert pairs_per_second > 100, \
            f"HMM throughput {pairs_per_second:.0f} pairs/s < 100 target"

    @pytest.mark.gpu
    def test_gpu_memory_bound(self):
        """GPU memory usage should stay within stated bounds for a given tile size."""
        import subprocess
        
        n, S = 200, 500_000
        G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
        positions = np.arange(S, dtype=np.float64) * 100
        
        fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
        
        # Record GPU memory before
        mem_before = gamma_smc_cu.gpu_memory_used()
        
        result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=1)
        
        mem_after = gamma_smc_cu.gpu_memory_used()
        mem_used_gb = (mem_after - mem_before) / 1e9
        
        # Should use less than 20 GB for this problem size
        assert mem_used_gb < 20.0, \
            f"GPU memory usage {mem_used_gb:.1f} GB exceeds 20 GB bound"

    @pytest.mark.gpu
    def test_scaling_linear_in_sites(self):
        """Runtime should scale approximately linearly with number of sites."""
        import time
        
        n = 50
        times = []
        site_counts = [50_000, 100_000, 200_000]
        
        for S in site_counts:
            G = np.random.randint(0, 2, size=(n, S), dtype=np.uint8)
            positions = np.arange(S, dtype=np.float64) * 100
            fc = gamma_smc_cu.CoalescenceEstimator(G, positions, mu=1.25e-8, rho=1e-8)
            
            start = time.perf_counter()
            fc.infer_tmrca(pairs=[(0, 1)], max_iterations=1)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Ratio of times should be roughly proportional to ratio of sites
        ratio_sites = site_counts[-1] / site_counts[0]
        ratio_time = times[-1] / times[0]
        
        # Allow 2x overhead for non-linear components
        assert ratio_time < ratio_sites * 2.0, \
            f"Scaling superlinear: {ratio_sites}x sites → {ratio_time:.1f}x time"
```

### B.9 Reference Implementations

Each reference implementation is a pure-numpy version of the corresponding CUDA kernel, written for clarity rather than performance. These are the source of truth for correctness testing.

```
tests/reference/
├── hmm_numpy.py          # NumpyHMM class: forward, backward, posterior, log_likelihood
│                          #   - Explicit loops over sites and time bins
│                          #   - Full FP64 precision
│                          #   - Validates: alpha, beta, gamma, log-likelihood
│
├── ultrametric_numpy.py   # NumpyUltrametric class: project, generate_noisy_tree
│                          #   - Agglomerative clustering with posterior scoring
│                          #   - Ultrametric violation metric
│                          #   - Tree simulation for test fixtures
│
├── pelt_numpy.py          # NumpyPELT class: detect changepoints
│                          #   - Standard PELT with Poisson cost function
│                          #   - BIC penalty
│                          #   - Returns list of breakpoint positions
│
└── sfs_numpy.py           # compute_sfs_numpy: column-wise allele counting
                           #   - Simple loop, no bitpacking
                           #   - Returns unfolded SFS array
```

### B.10 Test Data Management

**Simulated test data** is generated on-the-fly by msprime fixtures with fixed random seeds. This ensures reproducibility without committing large data files.

**Golden reference files** are generated from validated runs and committed to `tests/regression/golden/`. They are small (~1 MB) numpy archives. Regeneration command:

```bash
python tests/regression/generate_golden.py
```

**Real data for benchmarks** (1000 Genomes, Ag1000G) is not committed. Benchmark scripts download or expect paths via environment variables:

```bash
export TMRCA_CU_1KG_PATH=/data/1000genomes/
export TMRCA_CU_AG1000G_PATH=/data/ag1000g/
pytest benchmarks/ -v
```

---

## Appendix C: Plotting (`python/gamma_smc_cu/plotting.py`)

Minimal, Nature-style publication figures for coalescence time fields. Design principles: high data-ink ratio, no chartjunk, colorblind-safe, Helvetica/Arial, 300 DPI, thin spines, no top/right axes, lowercase bold panel labels.

This file lives at `python/gamma_smc_cu/plotting.py` and is the only visualization dependency in the package.

```python
"""
gamma_smc_cu — Minimal Nature-style plotting.

Produces publication-ready figures for coalescence time fields.
Design: clean, minimal, high data-ink ratio. No chartjunk.

Fonts: Helvetica/Arial. Colors: muted, colorblind-safe.
Panel labels: lowercase bold (a, b, c). No gridlines. Thin spines.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# ── Nature style ──────────────────────────────────────────────

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Muted palette — colorblind-safe, print-friendly
C = {
    "true":     "#2d2d2d",
    "est":      "#c44e52",
    "ci":       "#c44e52",
    "pi":       "#4c72b0",
    "prior":    "#cccccc",
    "accent1":  "#dd8452",
    "accent2":  "#55a868",
    "accent3":  "#8172b3",
}

# TMRCA heatmap: dark (recent) → light (ancient)
TMRCA_CMAP = LinearSegmentedColormap.from_list(
    "tmrca", ["#1a1a2e", "#16213e", "#0f3460", "#c44e52", "#e8d5b7"], N=256
)


def _panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def _format_mb(x, _):
    return f"{x / 1e6:.1f}"


def plot_tmrca_landscape(
    positions,
    true_tmrca=None,
    est_tmrca=None,
    est_lower=None,
    est_upper=None,
    pi=None,
    posterior=None,
    time_bins=None,
    pair_label="(0, 1)",
    figsize=(183/25.4, 120/25.4),  # Nature single-column: 183 mm
    output_path=None,
):
    """
    Main figure: TMRCA landscape with up to 4 panels.

    a) TMRCA trace — true vs estimated with credible interval
    b) Posterior heatmap — P(T|data) as image
    c) Per-site π
    d) Posterior marginal at focal sites

    Panels are included only if the corresponding data is provided.

    Parameters
    ----------
    positions : np.ndarray, shape (S,)
        Physical positions of segregating sites (bp).
    true_tmrca : np.ndarray, shape (S,), optional
        True TMRCA at each site (from simulation).
    est_tmrca : np.ndarray, shape (S,), optional
        Estimated posterior mean TMRCA.
    est_lower, est_upper : np.ndarray, shape (S,), optional
        Lower and upper bounds of 95% credible interval.
    pi : np.ndarray, shape (S,), optional
        Per-site nucleotide diversity.
    posterior : np.ndarray, shape (S, K), optional
        Full posterior marginals at each site.
    time_bins : np.ndarray, shape (K,), optional
        Midpoints of time discretization bins (generations).
    pair_label : str
        Label for the haplotype pair shown.
    figsize : tuple
        Figure size in inches. Default is Nature single-column width.
    output_path : str, optional
        Path to save figure (supports .png, .pdf, .svg).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.rcParams.update(NATURE_RC)

    # Determine which panels to show
    panels = []
    if true_tmrca is not None or est_tmrca is not None:
        panels.append("tmrca")
    if posterior is not None and time_bins is not None:
        panels.append("heatmap")
    if pi is not None:
        panels.append("pi")
    if posterior is not None and time_bins is not None:
        panels.append("marginal")

    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("No data provided for any panel")

    height_ratios = []
    for p in panels:
        if p == "heatmap":
            height_ratios.append(1.2)
        elif p == "marginal":
            height_ratios.append(0.8)
        else:
            height_ratios.append(1.0)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.45},
        squeeze=False,
    )
    axes = axes.flatten()
    labels = "abcdefgh"
    pos_mb = positions

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        _panel_label(ax, labels[idx])

        if panel == "tmrca":
            if est_tmrca is not None and est_lower is not None and est_upper is not None:
                ax.fill_between(
                    pos_mb, est_lower, est_upper,
                    color=C["ci"], alpha=0.15, linewidth=0, label="95% CI",
                )
            if true_tmrca is not None:
                ax.plot(pos_mb, true_tmrca, color=C["true"], linewidth=0.6,
                        label="True $T_\\mathrm{MRCA}$", zorder=3)
            if est_tmrca is not None:
                ax.plot(pos_mb, est_tmrca, color=C["est"], linewidth=0.5,
                        alpha=0.85, label="Estimated", zorder=2)
            ax.set_ylabel("$T_\\mathrm{MRCA}$ (gen)")
            ax.set_yscale("log")
            ax.legend(loc="upper right", frameon=False, ncol=3, columnspacing=1)
            ax.set_title(f"Pairwise coalescence time — pair {pair_label}",
                        fontsize=8, loc="left", pad=4)

        elif panel == "heatmap":
            max_pixels = 2000
            if posterior.shape[0] > max_pixels:
                idx_sub = np.linspace(0, posterior.shape[0] - 1, max_pixels).astype(int)
                post_img = posterior[idx_sub].T
                pos_sub = pos_mb[idx_sub]
            else:
                post_img = posterior.T
                pos_sub = pos_mb
            extent = [pos_sub[0], pos_sub[-1], 0, len(time_bins) - 1]
            im = ax.imshow(
                post_img, aspect="auto", origin="lower",
                extent=extent, cmap=TMRCA_CMAP, interpolation="bilinear",
                vmin=0, vmax=np.percentile(post_img, 99),
            )
            n_yticks = 6
            ytick_idx = np.linspace(0, len(time_bins) - 1, n_yticks).astype(int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f"{time_bins[i]:.0f}" for i in ytick_idx])
            ax.set_ylabel("$T$ (gen)")
            cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=20, pad=0.02)
            cbar.ax.set_ylabel("$P(T|\\mathrm{data})$", fontsize=6)
            cbar.ax.tick_params(labelsize=5)
            cbar.outline.set_linewidth(0.3)

        elif panel == "pi":
            window = min(100, len(pi) // 10)
            if window > 1:
                kernel = np.ones(window) / window
                pi_smooth = np.convolve(pi, kernel, mode="same")
            else:
                pi_smooth = pi
            ax.fill_between(pos_mb, 0, pi_smooth, color=C["pi"], alpha=0.3, linewidth=0)
            ax.plot(pos_mb, pi_smooth, color=C["pi"], linewidth=0.5)
            ax.set_ylabel("$\\pi$")
            ax.set_ylim(bottom=0)

        elif panel == "marginal":
            n_sites = posterior.shape[0]
            focal_sites = [n_sites // 4, n_sites // 2, 3 * n_sites // 4]
            colors_marginal = [C["est"], C["accent1"], C["accent3"]]
            for site_idx, col in zip(focal_sites, colors_marginal):
                pos_label = f"{pos_mb[site_idx]/1e6:.2f} Mb"
                ax.fill_between(
                    time_bins, 0, posterior[site_idx],
                    alpha=0.25, color=col, linewidth=0,
                )
                ax.plot(time_bins, posterior[site_idx], color=col,
                       linewidth=0.7, label=pos_label)
            ax.set_xlabel("$T$ (generations)")
            ax.set_ylabel("$P(T|\\mathrm{data})$")
            ax.set_xscale("log")
            ax.legend(loc="upper right", frameon=False, title="Focal sites",
                     title_fontsize=6)

        # Common x-axis formatting for genomic panels
        if panel != "marginal":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_mb))
            if idx == n_panels - 1 or (idx == n_panels - 2 and panels[-1] == "marginal"):
                ax.set_xlabel("Position (Mb)")
            else:
                ax.set_xticklabels([])

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.close(fig)
    return fig
```

### C.1 Usage with `gamma_smc_cu` Output

```python
import gamma_smc_cu
from gamma_smc_cu.plotting import plot_tmrca_landscape

# Run inference
fc = gamma_smc_cu.CoalescenceEstimator.from_tree_sequence(ts, gpu_ids=[0])
result = fc.infer_tmrca(pairs=[(0, 1)], max_iterations=5, subsample_size=50)

# Plot
fig = plot_tmrca_landscape(
    positions=fc.positions,
    true_tmrca=true_tmrca,           # from simulation, if available
    est_tmrca=result.tmrca_mean[0],
    est_lower=result.tmrca_lower[0],
    est_upper=result.tmrca_upper[0],
    pi=fc.site_pi(),
    posterior=result.gamma[0],       # (S, K) posterior marginals
    time_bins=result.time_midpoints,
    pair_label="(0, 1)",
    output_path="figure1.pdf",
)
```

### C.2 Demo Figure Generation

```python
# Generate demo figure from simulated data (no GPU required):
python -m gamma_smc_cu.plotting
```

This produces a 4-panel figure:
- **a)** TMRCA trace with true (black) vs estimated (red) and 95% CI shading
- **b)** Posterior heatmap $P(T|\text{data})$ across sites and time bins
- **c)** Per-site nucleotide diversity $\pi$ (smoothed)
- **d)** Posterior marginal distributions at three focal sites


