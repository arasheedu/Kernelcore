// Y = W X for GGUF Q6_K weights and several 8-bit activation columns: one thread computes one row
// for a tile of 8 columns, reading each weight word once per tile. Weight stream, scratch layout,
// dims and base as in matmul_q4k_q8x4.cu / matvec_q6k_q8.cu (tiles of 8 here).
// Each column's arithmetic is exactly matvec_q6k_q8's (the four blocks of a half super-block are
// summed left to right in p, then acc += d * p), so results are bit-identical to it.

#ifndef PCUDA_DP4A_DEFINED
#define PCUDA_DP4A_DEFINED
// u8 x s8 dot product of 4 packed bytes, plus c. The bytes of `a` must be 0..127 so that the
// GPU's signed __dp4a gives the same result. pcuda_mslgen maps this helper to AVX-VNNI.
__device__ int pcuda_dp4a(int a, int b, int c)
{
#if defined(__NVCC__)
    return __dp4a(a, b, c);
#else
    c += (a & 255) * ((b << 24) >> 24);
    c += ((a >> 8) & 255) * ((b << 16) >> 24);
    c += ((a >> 16) & 255) * ((b << 8) >> 24);
    c += ((a >> 24) & 255) * (b >> 24);
    return c;
#endif
}
#endif

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void matmul_q6k_q8x8(const int* qw, const int* xq, float* y, const int* dims, int base)
{
    const int idx = pcuda_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = pcuda_aligned8(dims[0]);
    const int k = dims[1];
    const int ncols = dims[2];
    const int rb = dims[3];
    const int tiles = (ncols + 7) / 8;
    if (idx < rows * tiles) {
        // Blocks of rb 8-row groups; each block runs its groups for tile 0, then tile 1, ...: the
        // block's weights stay in L2 across tiles and a tile's activations in L1 across groups.
        const int ng = rows / 8;
        const int grp = idx / 8;
        const int blk = grp / (rb * tiles);
        const int g0 = blk * rb;
        int rbn;
        if (ng - g0 < rb) rbn = ng - g0;
        else rbn = rb;
        const int within = grp - g0 * tiles;
        const int tile = within / rbn;
        // g and l directly (not row / 8): g stays uniform across a simd group and l = lane + uniform,
        // so pcuda_mslgen loads weight words as consecutive lanes (no per-load index checks).
        const int g = g0 + within - tile * rbn;
        const int l = idx - grp * 8;
        const int row = g * 8 + l;
        const int nsb = k / 256;
        const int nb = k / 32;
        const int c0 = tile * 8;
        int c1;
        if (c0 + 1 < ncols) c1 = c0 + 1;
        else c1 = c0;
        int c2;
        if (c0 + 2 < ncols) c2 = c0 + 2;
        else c2 = c1;
        int c3;
        if (c0 + 3 < ncols) c3 = c0 + 3;
        else c3 = c2;
        int c4;
        if (c0 + 4 < ncols) c4 = c0 + 4;
        else c4 = c3;
        int c5;
        if (c0 + 5 < ncols) c5 = c0 + 5;
        else c5 = c4;
        int c6;
        if (c0 + 6 < ncols) c6 = c0 + 6;
        else c6 = c5;
        int c7;
        if (c0 + 7 < ncols) c7 = c0 + 7;
        else c7 = c6;
        const int * x0 = xq + c0 * nb * 11;
        const int * x1 = xq + c1 * nb * 11;
        const int * x2 = xq + c2 * nb * 11;
        const int * x3 = xq + c3 * nb * 11;
        const int * x4 = xq + c4 * nb * 11;
        const int * x5 = xq + c5 * nb * 11;
        const int * x6 = xq + c6 * nb * 11;
        const int * x7 = xq + c7 * nb * 11;
        const float * e0 = (const float *)(x0 + nb * 8);
        const float * e1 = (const float *)(x1 + nb * 8);
        const float * e2 = (const float *)(x2 + nb * 8);
        const float * e3 = (const float *)(x3 + nb * 8);
        const float * e4 = (const float *)(x4 + nb * 8);
        const float * e5 = (const float *)(x5 + nb * 8);
        const float * e6 = (const float *)(x6 + nb * 8);
        const float * e7 = (const float *)(x7 + nb * 8);
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int wb = (g * nsb + s) * 424;  // 53 words x 8 rows
            const float d = ((const float*)qw)[wb + 52 * 8 + l];
            for (int n = 0; n < 2; ++n) {
                float p0 = 0.0f;
                float p1 = 0.0f;
                float p2 = 0.0f;
                float p3 = 0.0f;
                float p4 = 0.0f;
                float p5 = 0.0f;
                float p6 = 0.0f;
                float p7 = 0.0f;
                for (int t = 0; t < 4; ++t) {
                    const int b = s * 8 + n * 4 + t;
                    const int qlw = 16 * n + 8 * (t & 1);
                    const int lowsh = 4 * (t >> 1);
                    const int hish = 2 * t;
                    const int scw = qw[wb + (48 + 2 * n + (t >> 1)) * 8 + l];
                    const int sh = 16 * (t & 1);
                    const int sca = (scw << (24 - sh)) >> 24;
                    const int scb = (scw << (16 - sh)) >> 24;
                    int da0 = 0;
                    int da1 = 0;
                    int da2 = 0;
                    int da3 = 0;
                    int da4 = 0;
                    int da5 = 0;
                    int da6 = 0;
                    int da7 = 0;
                    int db0 = 0;
                    int db1 = 0;
                    int db2 = 0;
                    int db3 = 0;
                    int db4 = 0;
                    int db5 = 0;
                    int db6 = 0;
                    int db7 = 0;
                    int sa0 = 0;
                    int sa1 = 0;
                    int sa2 = 0;
                    int sa3 = 0;
                    int sa4 = 0;
                    int sa5 = 0;
                    int sa6 = 0;
                    int sa7 = 0;
                    for (int w = 0; w < 4; ++w) {
                        const int lo = (qw[wb + (qlw + w) * 8 + l] >> lowsh) & 252645135;                  // 0x0F0F0F0F
                        const int hi = ((qw[wb + (32 + 8 * n + w) * 8 + l] >> hish) << 4) & 808464432;    // 0x30303030
                        const int q = lo | hi;
                        da0 = pcuda_dp4a(q, x0[b * 8 + w], da0);
                        da1 = pcuda_dp4a(q, x1[b * 8 + w], da1);
                        da2 = pcuda_dp4a(q, x2[b * 8 + w], da2);
                        da3 = pcuda_dp4a(q, x3[b * 8 + w], da3);
                        da4 = pcuda_dp4a(q, x4[b * 8 + w], da4);
                        da5 = pcuda_dp4a(q, x5[b * 8 + w], da5);
                        da6 = pcuda_dp4a(q, x6[b * 8 + w], da6);
                        da7 = pcuda_dp4a(q, x7[b * 8 + w], da7);
                        sa0 = pcuda_dp4a(16843009, x0[b * 8 + w], sa0);
                        sa1 = pcuda_dp4a(16843009, x1[b * 8 + w], sa1);
                        sa2 = pcuda_dp4a(16843009, x2[b * 8 + w], sa2);
                        sa3 = pcuda_dp4a(16843009, x3[b * 8 + w], sa3);
                        sa4 = pcuda_dp4a(16843009, x4[b * 8 + w], sa4);
                        sa5 = pcuda_dp4a(16843009, x5[b * 8 + w], sa5);
                        sa6 = pcuda_dp4a(16843009, x6[b * 8 + w], sa6);
                        sa7 = pcuda_dp4a(16843009, x7[b * 8 + w], sa7);
                    }
                    for (int w = 4; w < 8; ++w) {
                        const int lo = (qw[wb + (qlw + w) * 8 + l] >> lowsh) & 252645135;
                        const int hi = ((qw[wb + (32 + 8 * n + w) * 8 + l] >> hish) << 4) & 808464432;
                        const int q = lo | hi;
                        db0 = pcuda_dp4a(q, x0[b * 8 + w], db0);
                        db1 = pcuda_dp4a(q, x1[b * 8 + w], db1);
                        db2 = pcuda_dp4a(q, x2[b * 8 + w], db2);
                        db3 = pcuda_dp4a(q, x3[b * 8 + w], db3);
                        db4 = pcuda_dp4a(q, x4[b * 8 + w], db4);
                        db5 = pcuda_dp4a(q, x5[b * 8 + w], db5);
                        db6 = pcuda_dp4a(q, x6[b * 8 + w], db6);
                        db7 = pcuda_dp4a(q, x7[b * 8 + w], db7);
                    }
                    const int i0 = sca * (da0 - 32 * sa0) + scb * (db0 - 32 * ((int)e0[b * 2 + 1] - sa0));
                    const int i1 = sca * (da1 - 32 * sa1) + scb * (db1 - 32 * ((int)e1[b * 2 + 1] - sa1));
                    const int i2 = sca * (da2 - 32 * sa2) + scb * (db2 - 32 * ((int)e2[b * 2 + 1] - sa2));
                    const int i3 = sca * (da3 - 32 * sa3) + scb * (db3 - 32 * ((int)e3[b * 2 + 1] - sa3));
                    const int i4 = sca * (da4 - 32 * sa4) + scb * (db4 - 32 * ((int)e4[b * 2 + 1] - sa4));
                    const int i5 = sca * (da5 - 32 * sa5) + scb * (db5 - 32 * ((int)e5[b * 2 + 1] - sa5));
                    const int i6 = sca * (da6 - 32 * sa6) + scb * (db6 - 32 * ((int)e6[b * 2 + 1] - sa6));
                    const int i7 = sca * (da7 - 32 * sa7) + scb * (db7 - 32 * ((int)e7[b * 2 + 1] - sa7));
                    p0 += (float)i0 * e0[b * 2];
                    p1 += (float)i1 * e1[b * 2];
                    p2 += (float)i2 * e2[b * 2];
                    p3 += (float)i3 * e3[b * 2];
                    p4 += (float)i4 * e4[b * 2];
                    p5 += (float)i5 * e5[b * 2];
                    p6 += (float)i6 * e6[b * 2];
                    p7 += (float)i7 * e7[b * 2];
                }
                acc0 += d * p0;
                acc1 += d * p1;
                acc2 += d * p2;
                acc3 += d * p3;
                acc4 += d * p4;
                acc5 += d * p5;
                acc6 += d * p6;
                acc7 += d * p7;
            }
        }
        y[c0 * rows + row] = acc0;
        if (c0 + 1 < ncols) y[(c0 + 1) * rows + row] = acc1;
        if (c0 + 2 < ncols) y[(c0 + 2) * rows + row] = acc2;
        if (c0 + 3 < ncols) y[(c0 + 3) * rows + row] = acc3;
        if (c0 + 4 < ncols) y[(c0 + 4) * rows + row] = acc4;
        if (c0 + 5 < ncols) y[(c0 + 5) * rows + row] = acc5;
        if (c0 + 6 < ncols) y[(c0 + 6) * rows + row] = acc6;
        if (c0 + 7 < ncols) y[(c0 + 7) * rows + row] = acc7;
    }
}
