// y = W x for GGUF Q4_K weights (llama.cpp's block_q4_K) and 8-bit activations (quantize_q8x4).
// A Q4_K super-block holds 256 weights of one row: d, dmin, 12 bytes of packed 6-bit scales and
// mins for its 8 sub-blocks of 32, and 128 bytes of 4-bit values; weight = d*sc*q - dmin*m.
// Stream (ref::repackQ4K): rows in groups of 8, the 8 rows' super-blocks interleaved by 32-bit
// word, 37 words per row: [0] d (f32), [1] dmin (f32), [2..4] scales, [5..36] qs.
// Word w of row l of super-block s in group g is qw[((g * nsb + s) * 37 + w) * 8 + l].
// Activation block b (32 values, scale xd[2b], sum of q xd[2b+1]) is sub-block b % 8 of
// super-block b / 8. Per super-block:
//   d * sum_s(dx_s * sc_s * dot_s) - dmin * sum_s(m_s * dx_s * xsum_s)
// The 8 scales and 8 mins are unpacked once per super-block, four per word, with word-wide masks
// (ggml get_scale_min_k4: scales 0-3 / mins 0-3 are the low 6 bits of bytes 0-3 / 4-7; scales
// 4-7 and mins 4-7 take the nibbles of bytes 8-11 plus the top 2 bits of bytes 0-3 / 4-7).
// rows % 8 == 0, k % 256 == 0, dims = {rows, k}.

#ifndef KC_DP4A_DEFINED
#define KC_DP4A_DEFINED
// u8 x s8 dot product of 4 packed bytes, plus c. The bytes of `a` must be 0..127 so that the
// GPU's signed __dp4a gives the same result. kc_mslgen maps this helper to AVX-VNNI.
__device__ int kc_dp4a(int a, int b, int c)
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

extern "C" __global__ void matvec_q4k_q8(const int* qw, const int* xq, const float* xd, float* y,
                                         const int* dims)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = dims[0];
    const int k = dims[1];
    if (row < rows) {
        const int nsb = k / 256;
        const int g = row / 8;
        const int l = row - g * 8;
        float acc = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int base = (g * nsb + s) * 296;  // 37 words x 8 rows
            const float d = ((const float*)qw)[base + l];
            const float dmin = ((const float*)qw)[base + 8 + l];
            const int s0 = qw[base + 16 + l];  // scale bytes 0..3
            const int s1 = qw[base + 24 + l];  // 4..7
            const int s2 = qw[base + 32 + l];  // 8..11
            const int sclo = s0 & 1061109567;                                      // 0x3F3F3F3F: scales 0-3
            const int mlo = s1 & 1061109567;                                       // mins 0-3
            const int schi = (s2 & 252645135) | ((s0 >> 2) & 808464432);          // 0x0F0F0F0F, 0x30303030
            const int mhi = ((s2 >> 4) & 252645135) | ((s1 >> 2) & 808464432);    // scales / mins 4-7
            float fd = 0.0f;  // sum of dx * sc * dot
            float fm = 0.0f;  // sum of m * dx * xsum
            for (int j = 0; j < 4; ++j) {  // 64 weights: sub-blocks 2j (low nibbles), 2j+1 (high)
                const int b = s * 8 + j * 2;
                int da = 0;
                int db = 0;
                for (int w = 0; w < 8; ++w) {
                    const int q = qw[base + (5 + j * 8 + w) * 8 + l];
                    da = kc_dp4a(q & 252645135, xq[b * 8 + w], da);  // 0x0F0F0F0F
                    db = kc_dp4a((q >> 4) & 252645135, xq[b * 8 + 8 + w], db);
                }
                // scale and min bytes of sub-blocks 2j, 2j+1: in the low (j < 2) or high word
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
                fd += (float)(sca * da) * xd[b * 2] + (float)(scb * db) * xd[b * 2 + 2];
                fm += (float)ma * (xd[b * 2] * xd[b * 2 + 1]) + (float)mb * (xd[b * 2 + 2] * xd[b * 2 + 3]);
            }
            acc += d * fd - dmin * fm;
        }
        y[row] = acc;
    }
}
