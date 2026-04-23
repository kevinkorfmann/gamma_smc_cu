#include "gamma_smc_cu/hmm.h"
#include <cmath>

// Compute coalescent prior q[k] under constant population size Ne.
// Time bins use quadratic spacing: t_k = t_max * (k/K)^2
void compute_coalescent_prior(double Ne, double t_max,
                              double* time_boundaries,
                              double* time_midpoints,
                              double* coal_prior_out,
                              int K) {

    // Quadratic time discretization
    for (int k = 0; k <= K; k++) {
        double frac = (double)k / (double)K;
        time_boundaries[k] = t_max * frac * frac;
    }

    for (int k = 0; k < K; k++) {
        time_midpoints[k] = (time_boundaries[k] + time_boundaries[k + 1]) / 2.0;
    }

    // Coalescent prior: probability of coalescence in bin [t_k, t_{k+1})
    // Under constant diploid Ne: rate = 1/(2*Ne), so
    //   q_k = exp(-t_k / (2*Ne)) - exp(-t_{k+1} / (2*Ne))
    double two_Ne = 2.0 * Ne;
    double total = 0.0;
    for (int k = 0; k < K; k++) {
        double t_lo = time_boundaries[k];
        double t_hi = time_boundaries[k + 1];
        coal_prior_out[k] = exp(-t_lo / two_Ne) - exp(-t_hi / two_Ne);
        total += coal_prior_out[k];
    }

    // Normalize (should already sum to ~1 if t_max is large enough)
    if (total > 0.0) {
        for (int k = 0; k < K; k++) {
            coal_prior_out[k] /= total;
        }
    }
}
