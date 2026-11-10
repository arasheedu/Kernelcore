// matvec_q4x8_q8 with two rows per thread: thread t is lane l = t % 8 of the group pair
// p = t / 8 and computes row 16p + l (group 2p) and row 16p + 8 + l (group 2p + 1).
// Same weight stream as matvec_q4x8_q8; each row's arithmetic is identical to it, so results
// are bit-identical. Two independent weight streams and four dot-product chains per thread keep
// more loads in flight, and the 8-bit activations are read once for both rows.
// rows % 16 == 0. dims = {rows, k}; rows / 2 threads.

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

extern "C" __global__ void matvec_q4x8_q8x2(const int* qw, const int* xq, const float* xd, float* y,
                                            const int* dims)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = dims[0];
    const int k = dims[1];
    if (t < rows / 2) {
        const int nb = k / 32;
        const int p = t / 8;
        const int l = t - p * 8;
        const int g0 = p * 2;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const int base0 = (g0 * nb + b) * 40;
            const int base1 = base0 + nb * 40;
            int lo0 = 0;
            int hi0 = 0;
            int lo1 = 0;
            int hi1 = 0;
            for (int w = 0; w < 4; ++w) {
                const int q0 = qw[base0 + w * 8 + l];
                const int q1 = qw[base1 + w * 8 + l];
                const int xa = xq[b * 8 + w];
                const int xb = xq[b * 8 + 4 + w];
                lo0 = kc_dp4a(q0 & 252645135, xa, lo0);  // 0x0F0F0F0F
                hi0 = kc_dp4a((q0 >> 4) & 252645135, xb, hi0);
                lo1 = kc_dp4a(q1 & 252645135, xa, lo1);
                hi1 = kc_dp4a((q1 >> 4) & 252645135, xb, hi1);
            }
            const float* sc0 = (const float*)(qw + base0 + 32);
            const float* sc1 = (const float*)(qw + base1 + 32);
            const int corr = 8 * (int)xd[b * 2 + 1];
            const float dx = xd[b * 2];
            const int dot0 = lo0 + hi0 - corr;
            const int dot1 = lo1 + hi1 - corr;
            acc0 += (float)dot0 * (sc0[l] * dx);
            acc1 += (float)dot1 * (sc1[l] * dx);
        }
        y[g0 * 8 + l] = acc0;
        y[g0 * 8 + 8 + l] = acc1;
    }
}
