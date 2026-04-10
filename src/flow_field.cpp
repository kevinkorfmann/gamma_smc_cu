#include "tmrca_cu/flow_field.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

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

static float bilinear_f(const float* table, float mean_log10, float cv_log10,
                        float mean_min, float mean_step, float cv_min, float cv_step) {
    float fm = (mean_log10 - mean_min) / mean_step;
    float fc = (cv_log10 - cv_min) / cv_step;
    fm = std::max(0.0f, std::min(fm, (float)(FF_MEAN_N - 1)));
    fc = std::max(0.0f, std::min(fc, (float)(FF_CV_N - 1)));

    int m0 = (int)fm, c0 = (int)fc;
    if (m0 == FF_MEAN_N - 1) m0--;
    if (c0 == FF_CV_N - 1) c0--;
    float wm = fm - m0, wc = fc - c0;

    int base = m0 * FF_CV_N + c0;
    float v00 = table[base], v01 = table[base + 1];
    float v10 = table[base + FF_CV_N], v11 = table[base + FF_CV_N + 1];

    return (1 - wm) * ((1 - wc) * v00 + wc * v01)
         + wm * ((1 - wc) * v10 + wc * v11);
}

static constexpr double ENTROPY_MEAN_MAX_LOG10 = 4.0;
static constexpr double ENTROPY_MEAN_STEP_LOG10 = 0.0008;
static constexpr double ENTROPY_CV_MIN_LOG10 = -2.0;
static constexpr double ENTROPY_CV_MAX_LOG10 = 0.0;

static double digamma_approx(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    double inv4 = inv2 * inv2;
    double inv6 = inv4 * inv2;
    double inv8 = inv4 * inv4;
    double inv10 = inv8 * inv2;
    result += std::log(x) - 0.5 * inv
        - inv2 / 12.0
        + inv4 / 120.0
        - inv6 / 252.0
        + inv8 / 240.0
        - 5.0 * inv10 / 660.0;
    return result;
}

static double differential_entropy_gamma(double mean_log10, double cv_log10) {
    double alpha_log10 = -2.0 * cv_log10;
    double beta_log10 = alpha_log10 - mean_log10;
    double alpha = std::pow(10.0, alpha_log10);
    double beta = std::pow(10.0, beta_log10);
    return alpha - std::log(beta) + std::lgamma(alpha) + (1.0 - alpha) * digamma_approx(alpha);
}

static const std::vector<float>& entropy_cv_threshold_table() {
    static const std::vector<float> table = []() {
        int n = (int)std::llround(ENTROPY_MEAN_MAX_LOG10 / ENTROPY_MEAN_STEP_LOG10) + 1;
        std::vector<float> out((size_t)n, 0.0f);
        for (int i = 0; i < n; ++i) {
            double mean_log10 = i * ENTROPY_MEAN_STEP_LOG10;
            double low = ENTROPY_CV_MIN_LOG10;
            double high = ENTROPY_CV_MAX_LOG10;

            double h_low = differential_entropy_gamma(mean_log10, low);
            double h_high = differential_entropy_gamma(mean_log10, high);
            if (h_low > 1.0) {
                out[(size_t)i] = (float)low;
                continue;
            }
            if (h_high <= 1.0) {
                out[(size_t)i] = (float)high;
                continue;
            }

            for (int it = 0; it < 60; ++it) {
                double mid = 0.5 * (low + high);
                double h_mid = differential_entropy_gamma(mean_log10, mid);
                if (h_mid > 1.0) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            out[(size_t)i] = (float)high;
        }
        return out;
    }();
    return table;
}

static inline void clip_mc(float& m, float& c, float mmin, float mmax, float cmin, float cmax) {
    m = std::max(mmin, std::min(m, mmax));
    c = std::max(cmin, std::min(c, cmax));
    if (m >= 0.0f && m <= ENTROPY_MEAN_MAX_LOG10) {
        const auto& table = entropy_cv_threshold_table();
        float idx = m / (float)ENTROPY_MEAN_STEP_LOG10;
        int i0 = (int)idx;
        if (i0 >= (int)table.size() - 1) {
            c = std::min(c, table.back());
            return;
        }
        float w = idx - i0;
        float limit = (1.0f - w) * table[(size_t)i0] + w * table[(size_t)(i0 + 1)];
        c = std::min(c, limit);
        c = std::max(cmin, std::min(c, cmax));
    }
}

static inline void site_emission_f(
    bool is_het,
    float scaled_mu,
    float& m,
    float& c)
{
    float a_log = -2.0f * c;
    float b_log = a_log - m;
    // Upstream gamma_smc applies beta += mu at every observed site, and
    // het sites additionally apply alpha += 1.
    b_log = log10f(powf(10.0f, b_log) + scaled_mu);
    if (is_het) {
        a_log = log10f(powf(10.0f, a_log) + 1.0f);
    }
    m = a_log - b_log;
    c = -0.5f * a_log;
}

static inline void cache_lookup_f(
    const float* cache_mean,
    const float* cache_cv,
    int step_idx,
    float mean_log10,
    float cv_log10,
    float mean_min,
    float mean_step,
    float cv_min,
    float cv_step,
    float& out_m,
    float& out_c)
{
    const float* pm = cache_mean + (size_t)step_idx * FF_GRID;
    const float* pc = cache_cv + (size_t)step_idx * FF_GRID;
    out_m = bilinear_f(pm, mean_log10, cv_log10, mean_min, mean_step, cv_min, cv_step);
    out_c = bilinear_f(pc, mean_log10, cv_log10, mean_min, mean_step, cv_min, cv_step);
}

FlowFieldCache build_flow_field_cache(
    const FlowFieldData& ff,
    int n_max_steps,
    float scaled_rho_f,
    float scaled_mu_f)
{
    float scaled_rho = scaled_rho_f;
    float scaled_mu  = scaled_mu_f;

    float mean_min = ff.mean_log10_min;
    float mean_max = ff.mean_log10_max;
    float cv_min   = ff.cv_log10_min;
    float cv_max   = ff.cv_log10_max;
    float mean_step = (mean_max - mean_min) / (FF_MEAN_N - 1);
    float cv_step   = (cv_max - cv_min) / (FF_CV_N - 1);

    size_t total = (size_t)n_max_steps * FF_GRID;
    float* missing_mean = new float[total];
    float* missing_cv   = new float[total];
    float* cache_mean = new float[total];
    float* cache_cv   = new float[total];
    float* fwd_hom_site_mean = new float[total];
    float* fwd_hom_site_cv   = new float[total];
    float* fwd_het_site_mean = new float[total];
    float* fwd_het_site_cv   = new float[total];
    float* bwd_hom_site_mean = new float[total];
    float* bwd_hom_site_cv   = new float[total];
    float* bwd_het_site_mean = new float[total];
    float* bwd_het_site_cv   = new float[total];

    // Step 0: missing stretch = recombination only, no hom emission.
    for (int row = 0; row < FF_MEAN_N; row++) {
        for (int col = 0; col < FF_CV_N; col++) {
            float m = mean_min + row * mean_step;
            float c = cv_min + col * cv_step;

            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);
            float u = bilinear_f(ff.u, m, c, mean_min, mean_step, cv_min, cv_step);
            float v = bilinear_f(ff.v, m, c, mean_min, mean_step, cv_min, cv_step);
            m += u * scaled_rho;
            c += v * scaled_rho;
            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);

            int idx = row * FF_CV_N + col;
            missing_mean[idx] = (float)m;
            missing_cv[idx]   = (float)c;
        }
    }

    // Step 0: upstream gamma_smc treats a segment of length 1 as a single
    // recombination-only step followed by the observed site emission. The
    // hom-stretch cache therefore shares the same base step as the missing
    // cache; additional called positions are layered on in steps 1+ below.
    for (int row = 0; row < FF_MEAN_N; row++) {
        for (int col = 0; col < FF_CV_N; col++) {
            float m = mean_min + row * mean_step;
            float c = cv_min + col * cv_step;

            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);
            float u = bilinear_f(ff.u, m, c, mean_min, mean_step, cv_min, cv_step);
            float v = bilinear_f(ff.v, m, c, mean_min, mean_step, cv_min, cv_step);
            m += u * scaled_rho;
            c += v * scaled_rho;
            clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);

            int idx = row * FF_CV_N + col;  // step 0
            cache_mean[idx] = (float)m;
            cache_cv[idx]   = (float)c;
        }
    }

    // Steps 1..n_max_steps-1: compound from previous step.
    for (int step = 1; step < n_max_steps; step++) {
        size_t prev_off = (size_t)(step - 1) * FF_GRID;
        size_t cur_off  = (size_t)step * FF_GRID;

        for (int row = 0; row < FF_MEAN_N; row++) {
            for (int col = 0; col < FF_CV_N; col++) {
                int gidx = row * FF_CV_N + col;

                // Missing: recombination only.
                float m = missing_mean[prev_off + gidx];
                float c = missing_cv[prev_off + gidx];

                clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);
                float u = bilinear_f(ff.u, m, c, mean_min, mean_step, cv_min, cv_step);
                float v = bilinear_f(ff.v, m, c, mean_min, mean_step, cv_min, cv_step);
                m += u * scaled_rho;
                c += v * scaled_rho;
                clip_mc(m, c, mean_min, mean_max, cv_min, cv_max);

                missing_mean[cur_off + gidx] = (float)m;
                missing_cv[cur_off + gidx]   = (float)c;

                // Hom: previous step's result + another hom emission/recombination.
                float m_h = cache_mean[prev_off + gidx];
                float c_h = cache_cv[prev_off + gidx];

                // 1. Mutation emission
                float a_log = -2.0f * c_h;
                float b_log = a_log - m_h;
                float b_lin = powf(10.0f, b_log) + scaled_mu;
                b_log = log10f(b_lin);
                m_h = a_log - b_log;

                // 2. Recombination
                clip_mc(m_h, c_h, mean_min, mean_max, cv_min, cv_max);
                float u_h = bilinear_f(ff.u, m_h, c_h, mean_min, mean_step, cv_min, cv_step);
                float v_h = bilinear_f(ff.v, m_h, c_h, mean_min, mean_step, cv_min, cv_step);
                m_h += u_h * scaled_rho;
                c_h += v_h * scaled_rho;
                clip_mc(m_h, c_h, mean_min, mean_max, cv_min, cv_max);

                cache_mean[cur_off + gidx] = (float)m_h;
                cache_cv[cur_off + gidx]   = (float)c_h;
            }
        }
    }

    for (int step = 0; step < n_max_steps; step++) {
        size_t off = (size_t)step * FF_GRID;
        for (int row = 0; row < FF_MEAN_N; row++) {
            for (int col = 0; col < FF_CV_N; col++) {
                int gidx = row * FF_CV_N + col;

                // Forward: hom stretch of length step+1, then observed site.
                float m_f = cache_mean[off + gidx];
                float c_f = cache_cv[off + gidx];
                site_emission_f(false, scaled_mu, m_f, c_f);
                clip_mc(m_f, c_f, mean_min, mean_max, cv_min, cv_max);
                fwd_hom_site_mean[off + gidx] = (float)m_f;
                fwd_hom_site_cv[off + gidx] = (float)c_f;

                m_f = cache_mean[off + gidx];
                c_f = cache_cv[off + gidx];
                site_emission_f(true, scaled_mu, m_f, c_f);
                clip_mc(m_f, c_f, mean_min, mean_max, cv_min, cv_max);
                fwd_het_site_mean[off + gidx] = (float)m_f;
                fwd_het_site_cv[off + gidx] = (float)c_f;

                // Backward: observed site first, then hom stretch of length step+1.
                float m_b = mean_min + row * mean_step;
                float c_b = cv_min + col * cv_step;
                site_emission_f(false, scaled_mu, m_b, c_b);
                clip_mc(m_b, c_b, mean_min, mean_max, cv_min, cv_max);
                cache_lookup_f(
                    cache_mean, cache_cv, step,
                    m_b, c_b, mean_min, mean_step, cv_min, cv_step,
                    m_b, c_b
                );
                bwd_hom_site_mean[off + gidx] = (float)m_b;
                bwd_hom_site_cv[off + gidx] = (float)c_b;

                m_b = mean_min + row * mean_step;
                c_b = cv_min + col * cv_step;
                site_emission_f(true, scaled_mu, m_b, c_b);
                clip_mc(m_b, c_b, mean_min, mean_max, cv_min, cv_max);
                cache_lookup_f(
                    cache_mean, cache_cv, step,
                    m_b, c_b, mean_min, mean_step, cv_min, cv_step,
                    m_b, c_b
                );
                bwd_het_site_mean[off + gidx] = (float)m_b;
                bwd_het_site_cv[off + gidx] = (float)c_b;
            }
        }
    }

    return {
        n_max_steps,
        missing_mean,
        missing_cv,
        cache_mean,
        cache_cv,
        fwd_hom_site_mean,
        fwd_hom_site_cv,
        fwd_het_site_mean,
        fwd_het_site_cv,
        bwd_hom_site_mean,
        bwd_hom_site_cv,
        bwd_het_site_mean,
        bwd_het_site_cv,
    };
}

void free_flow_field_cache(FlowFieldCache& cache) {
    delete[] cache.missing_mean;
    delete[] cache.missing_cv;
    delete[] cache.mean;
    delete[] cache.cv;
    delete[] cache.fwd_hom_site_mean;
    delete[] cache.fwd_hom_site_cv;
    delete[] cache.fwd_het_site_mean;
    delete[] cache.fwd_het_site_cv;
    delete[] cache.bwd_hom_site_mean;
    delete[] cache.bwd_hom_site_cv;
    delete[] cache.bwd_het_site_mean;
    delete[] cache.bwd_het_site_cv;
    cache.missing_mean = nullptr;
    cache.missing_cv = nullptr;
    cache.mean = nullptr;
    cache.cv = nullptr;
    cache.fwd_hom_site_mean = nullptr;
    cache.fwd_hom_site_cv = nullptr;
    cache.fwd_het_site_mean = nullptr;
    cache.fwd_het_site_cv = nullptr;
    cache.bwd_hom_site_mean = nullptr;
    cache.bwd_hom_site_cv = nullptr;
    cache.bwd_het_site_mean = nullptr;
    cache.bwd_het_site_cv = nullptr;
    cache.n_max_steps = 0;
}
