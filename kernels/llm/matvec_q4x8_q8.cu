// y = W x with 4-bit weights and 8-bit activations (from quantize_q8x4), integer dot products.
// Weight stream, per group of 8 rows and per block of 32 weights: 40 ints =
//   32 words [4][8 rows]: word w of a row packs bytes 4w..4w+3 of the block; byte i holds
//                         weight 4w+i (low nibble) and weight 16+4w+i (high nibble), each +8
//   8 floats             the rows' block scales
// 160 bytes per 256 weights: the same 5 bits per weight as Q4. rows % 8 == 0. dims = {rows, k}.
// Per block: dot = sum(nibble * xq) over 32 weights, then (dot - 8 * sum(xq)) * d_w * d_x.

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

extern "C" __global__ void matvec_q4x8_q8(const int* qw, const int* xq, const float* xd, float* y,
                                          const int* dims)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = dims[0];
    const int k = dims[1];
    if (row < rows) {
        const int nb = k / 32;
        const int g = row / 8;
        const int l = row - g * 8;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const int base = (g * nb + b) * 40;
            int lo = 0;
            int hi = 0;
            for (int w = 0; w < 4; ++w) {
                const int q = qw[base + w * 8 + l];
                lo = pcuda_dp4a(q & 252645135, xq[b * 8 + w], lo);           // 0x0F0F0F0F
                hi = pcuda_dp4a((q >> 4) & 252645135, xq[b * 8 + 4 + w], hi);
            }
            const float* sc = (const float*)(qw + base + 32);
            const int dot = lo + hi - 8 * (int)xd[b * 2 + 1];
            acc += (float)dot * (sc[l] * xd[b * 2]);
        }
        y[row] = acc;
    }
}
