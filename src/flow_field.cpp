#include "tmrca_cu/flow_field.h"
#include <cstdio>
#include <cmath>
#include <cstring>

bool load_flow_field(const char* path, FlowFieldData& out) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "flow_field: cannot open %s\n", path);
        return false;
    }

    // Line 1: mean grid (min_linear, max_linear, n_steps)
    float mean_min_lin, mean_max_lin;
    int mean_n;
    if (fscanf(f, "%f %f %d", &mean_min_lin, &mean_max_lin, &mean_n) != 3 ||
        mean_n != FF_MEAN_N) {
        fprintf(stderr, "flow_field: bad mean grid (got n=%d, expected %d)\n",
                mean_n, FF_MEAN_N);
        fclose(f);
        return false;
    }

    // Line 2: cv grid (min_linear, max_linear, n_steps)
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

    // Lines 3..53: U values (51 rows × 50 cols)
    for (int r = 0; r < FF_MEAN_N; r++) {
        for (int c = 0; c < FF_CV_N; c++) {
            if (fscanf(f, "%f", &out.u[r * FF_CV_N + c]) != 1) {
                fprintf(stderr, "flow_field: truncated U at (%d,%d)\n", r, c);
                fclose(f);
                return false;
            }
        }
    }

    // Lines 54..104: V values (51 rows × 50 cols)
    for (int r = 0; r < FF_MEAN_N; r++) {
        for (int c = 0; c < FF_CV_N; c++) {
            if (fscanf(f, "%f", &out.v[r * FF_CV_N + c]) != 1) {
                fprintf(stderr, "flow_field: truncated V at (%d,%d)\n", r, c);
                fclose(f);
                return false;
            }
        }
    }

    fclose(f);
    return true;
}
