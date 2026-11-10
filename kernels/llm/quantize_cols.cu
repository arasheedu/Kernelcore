// 8-bit quantization of several columns in one launch, exactly as quantize_q8x4 does each block:
// column c (k values at x + c * k) into its scratch slot at q + c * nb * 11
// ([nb * 8 words | nb * 2 floats {scale, sum} | nb floats]), nb = k / 32.
// One thread per (column, block). dims = {nb, ncols}; nb % 8 == 0. base: first thread (chunking).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void quantize_cols(const float* x, int* q, const int* dims, int base)
{
    const int idx = kc_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = kc_aligned8(dims[0]);
    const int ncols = dims[1];
    if (idx < nb * ncols) {
        const int c = idx / nb;
        const int b = idx - c * nb;
        const int o = c * nb * 32 + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fmaxf(x[o + i], 0.0f - x[o + i]));
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int * qc = q + c * nb * 11;
        float * xd = (float *)(qc + nb * 8);
        int sum = 0;
        for (int w = 0; w < 8; ++w) {
            int word = 0;
            for (int i = 0; i < 4; ++i) {
                const float t = x[o + w * 4 + i] * id;
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
            qc[b * 8 + w] = word;
        }
        xd[b * 2] = d;
        xd[b * 2 + 1] = (float)sum;
    }
}
