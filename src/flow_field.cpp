#include "tmrca_cu/flow_field.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

bool load_flow_field(const char* path, FlowFieldData& out) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "flow_field: cannot open %s\n", path);
        return false;
    }

    float mean_min_lin, mean_max_lin;
    int mean_n;
    if (fscanf(f, "%f %f %d", &mean_min_lin, &mean_max_lin, &mean_n) != 3 ||
        mean_n != FF_MEAN_N) {
        fprintf(stderr, "flow_field: bad mean grid (got n=%d, expected %d)\n",
                mean_n, FF_MEAN_N);
        fclose(f);
        return false;
    }

    float cv_min_lin, cv_max_lin;
    int cv_n;
    if (fscanf(f, "%f %f %d", &cv_min_lin, &cv_max_lin, &cv_n) != 3 ||
        cv_n != FF_CV_N) {
        fprintf(stderr, "flow_field: bad cv grid (got n=%d, expected %d)\n",
                cv_n, FF_CV_N);
        fclose(f);
        return false;
    }

    out.mean_log10_min = log10f(mean_min_lin);
    out.mean_log10_max = log10f(mean_max_lin);
    out.cv_log10_min   = log10f(cv_min_lin);
    out.cv_log10_max   = log10f(cv_max_lin);

    for (int r = 0; r < FF_MEAN_N; r++)
        for (int c = 0; c < FF_CV_N; c++)
            if (fscanf(f, "%f", &out.u[r * FF_CV_N + c]) != 1) {
                fclose(f); return false;
            }

    for (int r = 0; r < FF_MEAN_N; r++)
        for (int c = 0; c < FF_CV_N; c++)
            if (fscanf(f, "%f", &out.v[r * FF_CV_N + c]) != 1) {
                fclose(f); return false;
            }

    fclose(f);
    return true;
}

// ============================================================
// Multi-step cache building (CPU, double precision)
// ============================================================

static double bilinear_d(const float* table, double mean_log10, double cv_log10,
                          double mean_min, double mean_step, double cv_min, double cv_step) {
    double fm = (mean_log10 - mean_min) / mean_step;
    double fc = (cv_log10 - cv_min) / cv_step;
    fm = std::max(0.0, std::min(fm, (double)(FF_MEAN_N - 2)));
    fc = std::max(0.0, std::min(fc, (double)(FF_CV_N - 2)));

    int m0 = (int)fm, c0 = (int)fc;
    double wm = fm - m0, wc = fc - c0;

    int base = m0 * FF_CV_N + c0;
    double v00 = table[base], v01 = table[base + 1];
    double v10 = table[base + FF_CV_N], v11 = table[base + FF_CV_N + 1];

    return (1 - wm) * ((1 - wc) * v00 + wc * v01)
         + wm * ((1 - wc) * v10 + wc * v11);
}

static inline void clip_mc(double& m, double& c, double mmin, double mmax, double cmin, double cmax) {
    m = std::max(mmin, std::min(m, mmax));
    c = std::max(cmin, std::min(c, cmax));
}

FlowFieldCache build_flow_field_cache(
    const FlowFieldData& ff,
    int n_max_steps,
    float scaled_rho_f,
    float scaled_mu_f)
{
    double scaled_rho = scaled_rho_f;
    double scaled_mu  = scaled_mu_f;

    double mean_min = ff.mean_log10_min;
    double mean_max = ff.mean_log10_max;
    double cv_min   = ff.cv_log10_min;
    double cv_max   = ff.cv_log10_max;
    double mean_step = (mean_max - mean_min) / (FF_MEAN_N - 1);
    double cv_step   = (cv_max - cv_min) / (FF_CV_N - 1);

    size_t total = (size_t)n_max_steps * FF_GRID;
    float* cache_mean = new float[total];
    float* cache_cv   = new float[total];

    // Step 0: apply mutation + recombination once from each grid point
    for (int row = 0; row < FF_MEAN_N; row++) {
        for (int col = 0; col < FF_CV_N; col++) {
            double m = mean_min + row * mean_step;
            double c = cv_min + col * cv_step;

            // 1. Mutation emission: beta += scaled_mu
            double a_log = -2.0 * c;
            double b_log = a_log - m;
            double b_lin = pow(10.0, b_log) + scaled_mu;
            b_log = log10(b_lin);
            m = a_log - b_log;
            // c unchanged

            // 2. Recombination via flow field
            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);
            double u = bilinear_d(ff.u, m, c, mean_min, mean_step, cv_min, cv_step);
            double v = bilinear_d(ff.v, m, c, mean_min, mean_step, cv_min, cv_step);
            m += u * scaled_rho;
            c += v * scaled_rho;
            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);

            int idx = row * FF_CV_N + col;  // step 0
            cache_mean[idx] = (float)m;
            cache_cv[idx]   = (float)c;
        }
    }

    // Steps 1..n_max_steps-1: compound from previous step
    for (int step = 1; step < n_max_steps; step++) {
        size_t prev_off = (size_t)(step - 1) * FF_GRID;
        size_t cur_off  = (size_t)step * FF_GRID;

        for (int row = 0; row < FF_MEAN_N; row++) {
            for (int col = 0; col < FF_CV_N; col++) {
                int gidx = row * FF_CV_N + col;

                // Start from previous step's result
                double m = cache_mean[prev_off + gidx];
                double c = cache_cv[prev_off + gidx];

                // 1. Mutation emission
                double a_log = -2.0 * c;
                double b_log = a_log - m;
                double b_lin = pow(10.0, b_log) + scaled_mu;
                b_log = log10(b_lin);
                m = a_log - b_log;

                // 2. Recombination
                clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);
                double u = bilinear_d(ff.u, m, c, mean_min, mean_step, cv_min, cv_step);
                double v = bilinear_d(ff.v, m, c, mean_min, mean_step, cv_min, cv_step);
                m += u * scaled_rho;
                c += v * scaled_rho;
                clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);

                cache_mean[cur_off + gidx] = (float)m;
                cache_cv[cur_off + gidx]   = (float)c;
            }
        }
    }

    return {n_max_steps, cache_mean, cache_cv};
}

void free_flow_field_cache(FlowFieldCache& cache) {
    delete[] cache.mean;
    delete[] cache.cv;
    cache.mean = nullptr;
    cache.cv = nullptr;
    cache.n_max_steps = 0;
}
