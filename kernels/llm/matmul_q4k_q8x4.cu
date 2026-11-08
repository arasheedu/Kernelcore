// Y = W X for GGUF Q4_K weights and several 8-bit activation columns (prompt processing, batched
// decoding): each thread computes one row for a tile of 4 columns, so every weight word is read
// once per 4 columns instead of once per column. Weight stream as matvec_q4k_q8.cu; each column's
// arithmetic is exactly matvec_q4k_q8's, so results are bit-identical to it column by column.
// xq: one scratch slot per column of nb = k / 32 blocks, [nb * 8 words | nb * 2 floats {scale,
// sum} | nb floats], column c at xq + c * nb * 11. y: column c at y + c * rows.
// One thread per (row, tile); a simd group is 8 rows of one tile, ordered in blocks of rb 8-row
// groups (see below; rb >= rows / 8 is plain tile-major order). rows % 8 == 0. A short last tile
// repeats its last column and does not store the repeats.
// dims = {rows, k, ncols, rb}; k % 256 == 0, rb >= 1. base: index of the first thread (chunking), else 0.

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

extern "C" __global__ void matmul_q4k_q8x4(const int* qw, const int* xq, float* y, const int* dims, int base)
{
    const int idx = pcuda_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = pcuda_aligned8(dims[0]);
    const int k = dims[1];
    const int ncols = dims[2];
    const int rb = dims[3];
    const int tiles = (ncols + 3) / 4;
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
        const int c0 = tile * 4;
        int c1;
        if (c0 + 1 < ncols) c1 = c0 + 1;
        else c1 = c0;
        int c2;
        if (c0 + 2 < ncols) c2 = c0 + 2;
        else c2 = c1;
        int c3;
        if (c0 + 3 < ncols) c3 = c0 + 3;
        else c3 = c2;
        const int * x0 = xq + c0 * nb * 11;
        const int * x1 = xq + c1 * nb * 11;
        const int * x2 = xq + c2 * nb * 11;
        const int * x3 = xq + c3 * nb * 11;
        const float * e0 = (const float *)(x0 + nb * 8);
        const float * e1 = (const float *)(x1 + nb * 8);
        const float * e2 = (const float *)(x2 + nb * 8);
        const float * e3 = (const float *)(x3 + nb * 8);
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int wb = (g * nsb + s) * 296;  // 37 words x 8 rows
            const float d = ((const float*)qw)[wb + l];
            const float dmin = ((const float*)qw)[wb + 8 + l];
            const int s0 = qw[wb + 16 + l];
            const int s1 = qw[wb + 24 + l];
            const int s2 = qw[wb + 32 + l];
            const int sclo = s0 & 1061109567;                                      // 0x3F3F3F3F
            const int mlo = s1 & 1061109567;
            const int schi = (s2 & 252645135) | ((s0 >> 2) & 808464432);          // 0x0F0F0F0F, 0x30303030
            const int mhi = ((s2 >> 4) & 252645135) | ((s1 >> 2) & 808464432);
            float fd0 = 0.0f;
            float fd1 = 0.0f;
            float fd2 = 0.0f;
            float fd3 = 0.0f;
            float fm0 = 0.0f;
            float fm1 = 0.0f;
            float fm2 = 0.0f;
            float fm3 = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const int b = s * 8 + j * 2;
                int da0 = 0;
                int da1 = 0;
                int da2 = 0;
                int da3 = 0;
                int db0 = 0;
                int db1 = 0;
                int db2 = 0;
                int db3 = 0;
                for (int w = 0; w < 8; ++w) {
                    const int q = qw[wb + (5 + j * 8 + w) * 8 + l];
                    const int lo = q & 252645135;
                    const int hi = (q >> 4) & 252645135;
                    da0 = pcuda_dp4a(lo, x0[b * 8 + w], da0);
                    db0 = pcuda_dp4a(hi, x0[b * 8 + 8 + w], db0);
                    da1 = pcuda_dp4a(lo, x1[b * 8 + w], da1);
                    db1 = pcuda_dp4a(hi, x1[b * 8 + 8 + w], db1);
                    da2 = pcuda_dp4a(lo, x2[b * 8 + w], da2);
                    db2 = pcuda_dp4a(hi, x2[b * 8 + 8 + w], db2);
                    da3 = pcuda_dp4a(lo, x3[b * 8 + w], da3);
                    db3 = pcuda_dp4a(hi, x3[b * 8 + 8 + w], db3);
                }
                const int sh = 16 * (j & 1);
                int scw;
                int mw;
                if (j < 2) {
                    scw = sclo;
                    mw = mlo;
                } else {
                    scw = schi;
                    mw = mhi;
                }
                const int sca = (scw >> sh) & 255;
                const int scb = (scw >> (sh + 8)) & 255;
                const int ma = (mw >> sh) & 255;
                const int mb = (mw >> (sh + 8)) & 255;
                fd0 += (float)(sca * da0) * e0[b * 2] + (float)(scb * db0) * e0[b * 2 + 2];
                fm0 += (float)ma * (e0[b * 2] * e0[b * 2 + 1]) + (float)mb * (e0[b * 2 + 2] * e0[b * 2 + 3]);
                fd1 += (float)(sca * da1) * e1[b * 2] + (float)(scb * db1) * e1[b * 2 + 2];
                fm1 += (float)ma * (e1[b * 2] * e1[b * 2 + 1]) + (float)mb * (e1[b * 2 + 2] * e1[b * 2 + 3]);
                fd2 += (float)(sca * da2) * e2[b * 2] + (float)(scb * db2) * e2[b * 2 + 2];
                fm2 += (float)ma * (e2[b * 2] * e2[b * 2 + 1]) + (float)mb * (e2[b * 2 + 2] * e2[b * 2 + 3]);
                fd3 += (float)(sca * da3) * e3[b * 2] + (float)(scb * db3) * e3[b * 2 + 2];
                fm3 += (float)ma * (e3[b * 2] * e3[b * 2 + 1]) + (float)mb * (e3[b * 2 + 2] * e3[b * 2 + 3]);
            }
            acc0 += d * fd0 - dmin * fm0;
            acc1 += d * fd1 - dmin * fm1;
            acc2 += d * fd2 - dmin * fm2;
            acc3 += d * fd3 - dmin * fm3;
        }
        y[c0 * rows + row] = acc0;
        if (c0 + 1 < ncols) y[(c0 + 1) * rows + row] = acc1;
        if (c0 + 2 < ncols) y[(c0 + 2) * rows + row] = acc2;
        if (c0 + 3 < ncols) y[(c0 + 3) * rows + row] = acc3;
    }
}
