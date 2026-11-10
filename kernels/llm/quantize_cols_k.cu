// 8-bit quantization of several columns with one scale per 256 values (as llama.cpp's Q8_K), for the
// matmul_*_q8kx* kernels: column c (k values at x + c * k) into its slot at q + c * nsb * 85,
// [nsb * 64 words (4 values each) | nsb floats (scale) | nsb * 16 ints (sums of 16 values) |
// nsb * 4 words: the 8 sums of 32 values B as bytes, B >> 6 for sums 0-3 and 4-7, then B & 63],
// nsb = k / 256. B = (B >> 6) * 64 + (B & 63) and |B| <= 4064, so both parts fit a signed byte:
// the Q4_K kernels take their min terms as dot products of these with the packed mins.
// Values round half away from zero, as quantize_q8x4 does, to -127..127.
// One thread per (column, super-block). dims = {nsb, ncols}. base: first thread (chunking).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void quantize_cols_k(const float* x, int* q, const int* dims, int base)
{
    const int idx = kc_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int nsb = dims[0];
    const int ncols = dims[1];
    if (idx < nsb * ncols) {
        const int c = idx / nsb;
        const int s = idx - c * nsb;
        const int o = c * nsb * 256 + s * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) amax = fmaxf(amax, fmaxf(x[o + i], 0.0f - x[o + i]));
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int * qc = q + c * nsb * 85;
        for (int g = 0; g < 16; ++g) {
            int sum = 0;
            for (int w = 0; w < 4; ++w) {
                int word = 0;
                for (int i = 0; i < 4; ++i) {
                    const float t = x[o + g * 16 + w * 4 + i] * id;
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
                qc[s * 64 + g * 4 + w] = word;
            }
            qc[nsb * 65 + s * 16 + g] = sum;
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
