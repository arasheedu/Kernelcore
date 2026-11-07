// y = W x for GGUF Q6_K weights (llama.cpp's block_q6_K) and 8-bit activations (quantize_q8x4).
// A Q6_K super-block holds 256 weights of one row: ql[128] (low 4 bits), qh[64] (high 2 bits),
// 16 signed 8-bit scales (one per 16 weights) and d; weight = d * scale * (q - 32), q 6-bit.
// Stream (ref::repackQ6K): rows in groups of 8, super-blocks interleaved by 32-bit word, 53 words
// per row: [0..31] ql, [32..47] qh, [48..51] scales, [52] d (f32).
// Word w of row l of super-block s in group g is qw[((g * nsb + s) * 53 + w) * 8 + l].
// Half n (128 weights) of a super-block is activation blocks 4n..4n+3. For word w (4 weights of
// each block), one ql word a = ql[16n + w], one ql word c = ql[16n + 8 + w] and one qh word h feed
// all four blocks: block 0 = a low nibbles + h bits 0-1, block 1 = c low + bits 2-3, block 2 =
// a high + bits 4-5, block 3 = c high + bits 6-7 (ggml dequantize_row_q6_K).
// Per 16 weights: scale * (dot(q, xq) - 32 * sum(xq)); the second half's sum is the block's sum
// (xd[2b+1]) minus the first half's.
// rows % 8 == 0, k % 256 == 0, dims = {rows, k}.

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

extern "C" __global__ void matvec_q6k_q8(const int* qw, const int* xq, const float* xd, float* y,
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
            const int base = (g * nsb + s) * 424;  // 53 words x 8 rows
            const float d = ((const float*)qw)[base + 52 * 8 + l];
            for (int n = 0; n < 2; ++n) {
                const int b = s * 8 + n * 4;  // activation blocks b..b+3
                // dot products of the four blocks' first (a) and second (b) 16 weights
                int d0a = 0;
                int d1a = 0;
                int d2a = 0;
                int d3a = 0;
                int d0b = 0;
                int d1b = 0;
                int d2b = 0;
                int d3b = 0;
                int s0a = 0;  // sums of the first 16 activations of each block
                int s1a = 0;
                int s2a = 0;
                int s3a = 0;
                for (int w = 0; w < 4; ++w) {
                    const int qa = qw[base + (16 * n + w) * 8 + l];
                    const int qc = qw[base + (16 * n + 8 + w) * 8 + l];
                    const int h = qw[base + (32 + 8 * n + w) * 8 + l];
                    const int x0 = xq[b * 8 + w];
                    const int x1 = xq[b * 8 + 8 + w];
                    const int x2 = xq[b * 8 + 16 + w];
                    const int x3 = xq[b * 8 + 24 + w];
                    d0a = pcuda_dp4a((qa & 252645135) | ((h << 4) & 808464432), x0, d0a);  // 0x0F0F0F0F, 0x30303030
                    d1a = pcuda_dp4a((qc & 252645135) | ((h << 2) & 808464432), x1, d1a);
                    d2a = pcuda_dp4a(((qa >> 4) & 252645135) | (h & 808464432), x2, d2a);
                    d3a = pcuda_dp4a(((qc >> 4) & 252645135) | ((h >> 2) & 808464432), x3, d3a);
                    s0a = pcuda_dp4a(16843009, x0, s0a);  // 0x01010101: sum of the 4 bytes
                    s1a = pcuda_dp4a(16843009, x1, s1a);
                    s2a = pcuda_dp4a(16843009, x2, s2a);
                    s3a = pcuda_dp4a(16843009, x3, s3a);
                }
                for (int w = 4; w < 8; ++w) {
                    const int qa = qw[base + (16 * n + w) * 8 + l];
                    const int qc = qw[base + (16 * n + 8 + w) * 8 + l];
                    const int h = qw[base + (32 + 8 * n + w) * 8 + l];
                    d0b = pcuda_dp4a((qa & 252645135) | ((h << 4) & 808464432), xq[b * 8 + w], d0b);
                    d1b = pcuda_dp4a((qc & 252645135) | ((h << 2) & 808464432), xq[b * 8 + 8 + w], d1b);
                    d2b = pcuda_dp4a(((qa >> 4) & 252645135) | (h & 808464432), xq[b * 8 + 16 + w], d2b);
                    d3b = pcuda_dp4a(((qc >> 4) & 252645135) | ((h >> 2) & 808464432), xq[b * 8 + 24 + w], d3b);
                }
                // signed scales: bytes 8n..8n+7 = words 48 + 2n (blocks 0, 1) and 49 + 2n (blocks 2, 3)
                const int sw0 = qw[base + (48 + 2 * n) * 8 + l];
                const int sw1 = qw[base + (49 + 2 * n) * 8 + l];
                const int s0b = (int)xd[b * 2 + 1] - s0a;
                const int s1b = (int)xd[b * 2 + 3] - s1a;
                const int s2b = (int)xd[b * 2 + 5] - s2a;
                const int s3b = (int)xd[b * 2 + 7] - s3a;
                const int i0 = ((sw0 << 24) >> 24) * (d0a - 32 * s0a) + ((sw0 << 16) >> 24) * (d0b - 32 * s0b);
                const int i1 = ((sw0 << 8) >> 24) * (d1a - 32 * s1a) + (sw0 >> 24) * (d1b - 32 * s1b);
                const int i2 = ((sw1 << 24) >> 24) * (d2a - 32 * s2a) + ((sw1 << 16) >> 24) * (d2b - 32 * s2b);
                const int i3 = ((sw1 << 8) >> 24) * (d3a - 32 * s3a) + (sw1 >> 24) * (d3b - 32 * s3b);
                acc += d * ((float)i0 * xd[b * 2] + (float)i1 * xd[b * 2 + 2] + (float)i2 * xd[b * 2 + 4] +
                            (float)i3 * xd[b * 2 + 6]);
            }
        }
        y[row] = acc;
    }
}
