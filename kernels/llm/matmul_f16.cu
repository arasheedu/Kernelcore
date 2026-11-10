// Y = W X for fp16 weights (ggml F16, e.g. Qwen3's attn_v) and fp32 activations.
// Stream (llm::ref::repackF16, at load time): rows in groups of 8, the 8 rows' 32-bit words (two
// fp16 each, the first value in the low half) interleaved: word w of row l of group g is
// qw[(g * (k / 2) + w) * 8 + l], so the 8 lanes of a group read 8 consecutive words.
// One thread per (4-column tile, row): each weight word is read once per 4 columns. Columns past
// the last repeat it (their sums are not stored).
// x: ncols columns of k floats; y: ncols columns of rows floats.
// dims = {rows (a multiple of 8), k (even), ncols}. base: index of the first thread (0 on the GPU;
// the chunk start under Kernelcore chunking).

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

extern "C" __global__ void matmul_f16(const int* qw, const float* x, float* y, const int* dims, float base)
{
    const int idx = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = kc_aligned8(dims[0]);
    const int k = dims[1];
    const int ncols = dims[2];
    const int ntiles = (ncols + 3) / 4;
    if (idx < rows * ntiles) {
        const int tile = idx / rows;
        const int row = idx - tile * rows;
        const int g = row / 8;
        const int l = row - g * 8;
        const int kw = k / 2;
        const int c0 = tile * 4;
        const int c1 = c0 + 1 < ncols ? c0 + 1 : c0;
        const int c2 = c0 + 2 < ncols ? c0 + 2 : c1;
        const int c3 = c0 + 3 < ncols ? c0 + 3 : c2;
        const int wb = g * kw * 8 + l;
        const int x0 = c0 * k;
        const int x1 = c1 * k;
        const int x2 = c2 * k;
        const int x3 = c3 * k;
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (int w = 0; w < kw; ++w) {
            const int q = qw[wb + w * 8];
            const float lo = kc_h2f(q & 65535);
            const float hi = kc_h2f((q >> 16) & 65535);
            a0 += lo * x[x0 + 2 * w] + hi * x[x0 + 2 * w + 1];
            a1 += lo * x[x1 + 2 * w] + hi * x[x1 + 2 * w + 1];
            a2 += lo * x[x2 + 2 * w] + hi * x[x2 + 2 * w + 1];
            a3 += lo * x[x3 + 2 * w] + hi * x[x3 + 2 * w + 1];
        }
        y[c0 * rows + row] = a0;
        if (c0 + 1 < ncols) y[c1 * rows + row] = a1;
        if (c0 + 2 < ncols) y[c2 * rows + row] = a2;
        if (c0 + 3 < ncols) y[c3 * rows + row] = a3;
    }
}
