// Attention output for ggml FLASH_ATTN_EXT (see fa_scores; p = fa_exp's softmax numerators):
// out[(i * H + h) * D + d] = sum over t of p[h][i][t] * V[t][h / group][d] / sum over t of p[h][i][t]
// One thread per (query i, kv head, d): it converts V[t][kvh][d] once per position and accumulates
// every query head of the group (in blocks of 4). V[t][kvh][d] is half (d & 1) of word
// t * v_row + kvh * v_head + d / 2. dims as in fa_scores. base: index of the first thread.

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

#ifndef PCUDA_H2F_DEFINED
#define PCUDA_H2F_DEFINED
// Value of the fp16 bit pattern in the low 16 bits of h. pcuda_mslgen's simd path maps calls to
// this exact body to F16C vcvtph2ps. The body itself (other paths) uses integer and float
// arithmetic only (Booth has no bit casts, and its x86 back end miscompiled a version with
// selects): exact for finite values; infinities and NaN become +-(1024 + mantissa) * 32 there.
__device__ float pcuda_h2f(int h)
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

extern "C" __global__ void fa_values(const float* p, const int* v, float* out, const int* dims, float base)
{
    const int idx = pcuda_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int H = dims[0];
    const int T = dims[1];
    const int D = pcuda_aligned8(dims[2]);
    const int group = dims[3];
    const int tp = dims[4];
    const int N = dims[9];
    const int hkv = dims[14];
    const int per = pcuda_aligned8(hkv * D);
    if (idx < N * per) {
        const int i = idx / per;
        const int r = idx - i * per;
        const int kvh = r / D;
        const int d = r - kvh * D;
        const int vc = kvh * dims[8] + d / 2;  // integer offsets, not pointer variables (pcuda_mslgen)
        const int sh = (d & 1) * 16;
        const int vr = dims[7];
        for (int g0 = 0; g0 < group; g0 += 4) {
            const int h0 = kvh * group + g0;
            const int ng = group - g0 < 4 ? group - g0 : 4;
            const int p0 = (h0 * N + i) * tp;
            const int ps = N * tp;
            // Branch-free (pcuda_mslgen): heads past the group's end repeat its last head.
            const int o1 = ng > 1 ? p0 + ps : p0;
            const int o2 = ng > 2 ? p0 + 2 * ps : o1;
            const int o3 = ng > 3 ? p0 + 3 * ps : o2;
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            for (int t = 0; t < T; ++t) {
                const int ah = (v[vc + t * vr] >> sh) & 65535;
                const float x = pcuda_h2f(ah);
                const float w0 = p[p0 + t];
                const float w1 = p[o1 + t];
                const float w2 = p[o2 + t];
                const float w3 = p[o3 + t];
                a0 += w0 * x;
                s0 += w0;
                a1 += w1 * x;
                s1 += w1;
                a2 += w2 * x;
                s2 += w2;
                a3 += w3 * x;
                s3 += w3;
            }
            const int o = (i * H + h0) * D + d;
            out[o] = a0 / s0;
            if (ng > 1) out[o + D] = a1 / s1;
            if (ng > 2) out[o + 2 * D] = a2 / s2;
            if (ng > 3) out[o + 3 * D] = a3 / s3;
        }
    }
}
