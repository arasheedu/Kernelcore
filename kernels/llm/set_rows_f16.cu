// ggml SET_ROWS into an fp16 tensor (the KV cache): dst row idx[r] = fp16(src row r).
// One thread per pair of values: it writes one 32-bit word (two fp16, the first in the low half).
// idx holds int64 row numbers; the low 32-bit word is used, and rows outside [0, dst rows) are skipped.
// The fp32 -> fp16 conversion rounds to nearest even, as F16C does, with float arithmetic only
// (Booth has no bit casts): values of 65520 and above become the largest finite fp16, and -0
// becomes +0. NaN gives an unspecified value.
// dims = {pairs per row (a multiple of 8), rows, src row stride (floats), dst row stride (words),
// dst rows}.
// base: index of the first thread (0 on the GPU; the chunk start under Kernelcore chunking).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

#ifndef KC_F2H_DEFINED
#define KC_F2H_DEFINED
// out = fp16 bit pattern (low 16 bits) of x, round to nearest even. A macro, not a function:
// kc_mslgen's simd path does not call helpers.
#define KC_F2H(out, xv)                                                                          \
    {                                                                                               \
        const float f2h_x = (xv);                                                                   \
        const int f2h_sign = f2h_x < 0.0f ? 32768 : 0;                                              \
        const float f2h_a0 = f2h_x < 0.0f ? -f2h_x : f2h_x;                                         \
        /* a in [1, 2) with a0 = a * 2^e, for a0 in [2^-14, 65520) */                              \
        float f2h_a = f2h_a0;                                                                       \
        int f2h_e = 0;                                                                              \
        bool f2h_c = f2h_a >= 256.0f;                                                               \
        f2h_a = f2h_c ? f2h_a * 0.00390625f : f2h_a;                                                \
        f2h_e += f2h_c ? 8 : 0;                                                                     \
        f2h_c = f2h_a >= 16.0f;                                                                     \
        f2h_a = f2h_c ? f2h_a * 0.0625f : f2h_a;                                                    \
        f2h_e += f2h_c ? 4 : 0;                                                                     \
        f2h_c = f2h_a >= 4.0f;                                                                      \
        f2h_a = f2h_c ? f2h_a * 0.25f : f2h_a;                                                      \
        f2h_e += f2h_c ? 2 : 0;                                                                     \
        f2h_c = f2h_a >= 2.0f;                                                                      \
        f2h_a = f2h_c ? f2h_a * 0.5f : f2h_a;                                                       \
        f2h_e += f2h_c ? 1 : 0;                                                                     \
        f2h_c = f2h_a < 0.0078125f;                                                                 \
        f2h_a = f2h_c ? f2h_a * 256.0f : f2h_a;                                                     \
        f2h_e -= f2h_c ? 8 : 0;                                                                     \
        f2h_c = f2h_a < 0.125f;                                                                     \
        f2h_a = f2h_c ? f2h_a * 16.0f : f2h_a;                                                      \
        f2h_e -= f2h_c ? 4 : 0;                                                                     \
        f2h_c = f2h_a < 0.5f;                                                                       \
        f2h_a = f2h_c ? f2h_a * 4.0f : f2h_a;                                                       \
        f2h_e -= f2h_c ? 2 : 0;                                                                     \
        f2h_c = f2h_a < 1.0f;                                                                       \
        f2h_a = f2h_c ? f2h_a * 2.0f : f2h_a;                                                       \
        f2h_e -= f2h_c ? 1 : 0;                                                                     \
        /* 12582912 = 1.5 * 2^23: adding and subtracting it rounds to an integer, ties to even */  \
        const float f2h_rn = ((f2h_a - 1.0f) * 1024.0f + 12582912.0f) - 12582912.0f;                \
        int f2h_bits = ((f2h_e + 15) << 10) + (int)f2h_rn; /* a mantissa of 1024 carries */         \
        const float f2h_rd = (f2h_a0 * 16777216.0f + 12582912.0f) - 12582912.0f; /* subnormal */    \
        f2h_bits = f2h_a0 < 6.103515625e-05f ? (int)f2h_rd : f2h_bits;                              \
        f2h_bits = f2h_a0 < 65520.0f ? f2h_bits : 31743;                                            \
        (out) = f2h_sign | f2h_bits;                                                                \
    }
#endif

extern "C" __global__ void set_rows_f16(const float* src, const int* idx, int* dst, const int* dims, float base)
{
    const int i = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = kc_aligned8(dims[0]);  // a multiple of 8
    const int rows = dims[1];
    if (i < pairs * rows) {
        const int r = i / pairs;
        const int j = i - r * pairs;
        const int row = idx[2 * r];
        const int so = r * dims[2] + 2 * j;
        int lo;
        int hi;
        KC_F2H(lo, src[so]);
        KC_F2H(hi, src[so + 1]);
        if ((unsigned)row < (unsigned)dims[4]) dst[row * dims[3] + j] = lo | (hi << 16);
    }
}
