// Attention scores for ggml FLASH_ATTN_EXT with an fp16 KV cache (kernels fa_scores, fa_max,
// fa_exp, fa_values run in that order; fa_values divides by the row sums).
// s[(h * N + i) * tp + t] = scale * dot(q[i][h], K[t][h / group]) + mask[i][t]
// One thread per (kv head, query i, position t): it converts K row t of its kv head once and scores
// it against every query head of the group (in blocks of 4). Positions t >= T are not written.
// K rows are fp16 pairs in 32-bit words: K[t][kvh][d] is half (d & 1) of word
// t * k_row + kvh * k_head + d / 2. The mask (if any) is fp16, row i at word i * m_row.
// dims = {H, T, D, group, tp, k_row, k_head, v_row, v_head, N, q_tok, q_head, m_row, has_mask, H_kv}
// (strides of q in floats, of k, v and the mask in words; D and tp multiples of 8).
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

#ifndef KC_H2F_DEFINED
#define KC_H2F_DEFINED
// Value of the fp16 bit pattern in the low 16 bits of h. kc_mslgen's simd path maps calls to
// this exact body to F16C vcvtph2ps. The body itself (other paths) uses integer and float
// arithmetic only (Booth has no bit casts, and its x86 back end miscompiled a version with
// selects): exact for finite values; infinities and NaN become +-(1024 + mantissa) * 32 there.
__device__ float kc_h2f(int h)
{
    const int e = (h >> 10) & 31;
    const int nz = (e + 31) >> 5;  // 1 for a normal number, 0 for a subnormal
    const int e1 = e + 1 - nz;     // a subnormal scales like e = 1
    const int ee = e1 - ((e1 + 1) >> 5);  // e = 31 (inf / NaN): use 30
    const float p = (1.0f + (float)(ee & 1)) * (1.0f + 3.0f * (float)((ee >> 1) & 1)) *
                    (1.0f + 15.0f * (float)((ee >> 2) & 1)) * (1.0f + 255.0f * (float)((ee >> 3) & 1)) *
                    (1.0f + 65535.0f * (float)((ee >> 4) & 1));  // 2^ee
    return (float)((h & 1023) + (nz << 10)) * p * 2.98023223876953125e-8f * (float)(1 - 2 * ((h >> 15) & 1));
}
#endif

extern "C" __global__ void fa_scores(const float* qv, const int* k, const int* mask, float* s, const int* dims,
                                     float scale, float base)
{
    const int idx = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int T = dims[1];
    const int D = dims[2];
    const int group = dims[3];
    const int tp = kc_aligned8(dims[4]);
    const int N = dims[9];
    const int per = kc_aligned8(N * tp);
    if (idx < dims[14] * per) {
        const int kvh = idx / per;
        const int r = idx - kvh * per;
        const int i = r / tp;
        const int t = r - i * tp;
        if (t < T) {
            // Without a mask (has = 0), read word 0 and drop it. Arithmetic, not selects: Booth kept
            // the first parameter in memory with selects here, which kc_mslgen cannot translate.
            const int has = dims[13];
            const int mw = mask[(i * dims[12] + t / 2) * has];  // has is 0 or 1
            const int mh = (mw >> ((t & 1) * 16)) & 65535;
            const float mm = kc_h2f(mh);
            const float mv = mm * (float)has;
            // Integer offsets, not pointer variables (kc_mslgen).
            const int kr = t * dims[5] + kvh * dims[6];
            const int qh = dims[11];
            for (int g0 = 0; g0 < group; g0 += 4) {
                const int h0 = kvh * group + g0;
                const int q0 = i * dims[10] + h0 * qh;
                const int ng = group - g0 < 4 ? group - g0 : 4;
                // Heads past the group's end repeat its last head (all four accumulators always run).
                const int o1 = ng > 1 ? q0 + qh : q0;
                const int o2 = ng > 2 ? q0 + 2 * qh : o1;
                const int o3 = ng > 3 ? q0 + 3 * qh : o2;
                float a0 = 0.0f;
                float a1 = 0.0f;
                float a2 = 0.0f;
                float a3 = 0.0f;
                for (int j = 0; j < D / 2; ++j) {
                    const int w = k[kr + j];
                    const int ah = w & 65535;
                    const float k0 = kc_h2f(ah);
                    const int bh = (w >> 16) & 65535;
                    const float k1 = kc_h2f(bh);
                    a0 += qv[q0 + 2 * j] * k0 + qv[q0 + 2 * j + 1] * k1;
                    a1 += qv[o1 + 2 * j] * k0 + qv[o1 + 2 * j + 1] * k1;
                    a2 += qv[o2 + 2 * j] * k0 + qv[o2 + 2 * j + 1] * k1;
                    a3 += qv[o3 + 2 * j] * k0 + qv[o3 + 2 * j + 1] * k1;
                }
                const int s0 = (h0 * N + i) * tp + t;
                const int ps = N * tp;
                s[s0] = a0 * scale + mv;
                if (ng > 1) s[s0 + ps] = a1 * scale + mv;
                if (ng > 2) s[s0 + 2 * ps] = a2 * scale + mv;
                if (ng > 3) s[s0 + 3 * ps] = a3 * scale + mv;
            }
        }
    }
}
