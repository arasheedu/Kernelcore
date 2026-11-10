// norm_mul_q8_f with the 8-bit output scaled per 256 values (quantize_cols_k's slot), for batches:
// y = x * r * w, r = 1 / sqrt(sum of the row's partial sums / n + eps), and each super-block of y
// quantized exactly as quantize_cols_k does it. One thread per (row, super-block of 256).
// q: one slot of nsb * 85 ints per row (row at q + row * nsb * 85, nsb = n / 256); the partial sums
// of x^2 (add_sumsq_f / sumsq_part, slots of (n / 32) * 11 ints) at q + dims[3].
// dims = {nsb, n, rows, offset of the partial-sum slots}. base: first thread as a float (chunking).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void norm_mul_q8k_f(const float* x, const float* w, float* y, int* q, const int* dims, float eps,
                                          float base)
{
    const int idx = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int nsb = dims[0];
    const int n = dims[1];
    const int rows = dims[2];
    if (idx < nsb * rows) {
        const int row = idx / nsb;
        const int s = idx - row * nsb;
        const int nb = nsb * 8;
        const float * ss = (const float *)(q + dims[3] + row * nb * 11 + nb * 10);
        float tot = 0.0f;
        for (int j = 0; j < nb; ++j) tot += ss[j];
        const float r = 1.0f / sqrtf(tot / (float)n + eps);
        const int o = row * n + s * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) {
            const float v = x[o + i] * r * w[s * 256 + i];
            y[o + i] = v;
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int * qc = q + row * nsb * 85;
        for (int h = 0; h < 16; ++h) {
            int sum = 0;
            for (int k = 0; k < 4; ++k) {
                int word = 0;
                for (int i = 0; i < 4; ++i) {
                    const float t = y[o + h * 16 + k * 4 + i] * id;
                    int qi;
                    if (t >= 0.0f) qi = (int)(t + 0.5f);
                    else qi = (int)(t - 0.5f);
                    int cl;
                    if (qi > 127) cl = 127;
                    else if (qi < -127) cl = -127;
                    else cl = qi;
                    sum += cl;
                    word = word | ((cl & 255) << (i * 8));
                }
                qc[s * 64 + h * 4 + k] = word;
            }
            qc[nsb * 65 + s * 16 + h] = sum;
        }
        int hlo = 0;
        int hhi = 0;
        int llo = 0;
        int lhi = 0;
        for (int i = 0; i < 4; ++i) {
            const int b0 = qc[nsb * 65 + s * 16 + i * 2] + qc[nsb * 65 + s * 16 + i * 2 + 1];
            const int b1 = qc[nsb * 65 + s * 16 + 8 + i * 2] + qc[nsb * 65 + s * 16 + 8 + i * 2 + 1];
            hlo = hlo | (((b0 >> 6) & 255) << (i * 8));
            hhi = hhi | (((b1 >> 6) & 255) << (i * 8));
            llo = llo | ((b0 & 63) << (i * 8));
            lhi = lhi | ((b1 & 63) << (i * 8));
        }
        qc[nsb * 81 + s * 4] = hlo;
        qc[nsb * 81 + s * 4 + 1] = hhi;
        qc[nsb * 81 + s * 4 + 2] = llo;
        qc[nsb * 81 + s * 4 + 3] = lhi;
        ((float *)(qc + nsb * 64))[s] = d;
    }
}
