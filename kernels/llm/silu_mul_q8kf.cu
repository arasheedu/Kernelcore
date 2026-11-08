// silu_mul_q8x4f with the 8-bit output scaled per 256 values (quantize_cols_k's slot), for batches:
// y = g / (1 + e^-g) * u, and each super-block of y quantized exactly as quantize_cols_k does it.
// One thread per (column, super-block of 256). Column c: y[c * n ...], slot q + c * nsb * 85.
// dims = {n, columns}; n % 256 == 0. base: first thread as a float (chunking).

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void silu_mul_q8kf(const float* g, const float* u, float* y, int* q, const int* dims, float base)
{
    const int idx = pcuda_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int n = dims[0];
    const int cols = dims[1];
    const int nsb = n / 256;
    if (idx < nsb * cols) {
        const int c = idx / nsb;
        const int s = idx - c * nsb;
        const int o = c * n + s * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) {
            const float v = g[o + i] / (1.0f + exp2f(-g[o + i] * 1.44269504f)) * u[o + i];
            y[o + i] = v;
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int * qc = q + c * nsb * 85;
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
