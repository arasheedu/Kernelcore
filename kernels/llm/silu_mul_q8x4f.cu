// SwiGLU with both outputs: y = g / (1 + e^-g) * u (as silu_mul computes it) and y quantized to
// 8 bits exactly as quantize_q8x4 does it, for the down projection. One thread per block of 32.
// Columns of n values: block b of column c reads y[c * n + 32 b ...] and writes column c's
// scratch slot q + c * (n / 32) * 11 ([nb * 8 words | nb * 2 floats {scale, sum} | nb floats]).
// dims = {n, columns}; n % 256 == 0. base: index of the first thread as a float (exact below 2^24;
// Booth takes at most 5 integer/pointer parameters), 0 when not chunked.

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void silu_mul_q8x4f(const float* g, const float* u, float* y, int* q, const int* dims, float base)
{
    const int idx = pcuda_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int n = dims[0];
    const int cols = dims[1];
    const int nb = pcuda_aligned8(n / 32);
    if (idx < nb * cols) {
        const int c = idx / nb;
        const int b = idx - c * nb;
        const int o = c * n + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = g[o + i] / (1.0f + exp2f(-g[o + i] * 1.44269504f)) * u[o + i];
            y[o + i] = v;
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int * qc = q + c * nb * 11;
        float * xd = (float *)(qc + nb * 8);
        int sum = 0;
        for (int k = 0; k < 8; ++k) {
            int word = 0;
            for (int i = 0; i < 4; ++i) {
                const float t = y[o + k * 4 + i] * id;
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
            qc[b * 8 + k] = word;
        }
        xd[b * 2] = d;
        xd[b * 2 + 1] = (float)sum;
    }
}
