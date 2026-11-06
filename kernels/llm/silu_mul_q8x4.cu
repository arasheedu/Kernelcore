// SwiGLU fused with 8-bit quantization for the next matvec_q4x8_q8 (one thread per block of 32):
//   h = g / (1 + e^-g) * u, then exactly what quantize_q8x4 does to h.
// Same float operations, in the same order, as silu_mul followed by quantize_q8x4; h itself is
// never written. Every conditional assignment has an explicit else (BOOTH_PIN F18).
extern "C" __global__ void silu_mul_q8x4(const float* g, const float* u, int* xq, float* xd, int count)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < count) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float gi = g[b * 32 + i];
            const float v = gi / (1.0f + exp2f(-gi * 1.44269504f)) * u[b * 32 + i];
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int sum = 0;
        for (int w = 0; w < 8; ++w) {
            int word = 0;
            for (int i = 0; i < 4; ++i) {
                const float gi = g[b * 32 + w * 4 + i];
                const float v = gi / (1.0f + exp2f(-gi * 1.44269504f)) * u[b * 32 + w * 4 + i];
                const float t = v * id;
                int q;
                if (t >= 0.0f) q = (int)(t + 0.5f);
                else q = (int)(t - 0.5f);
                int c;
                if (q > 127) c = 127;
                else if (q < -127) c = -127;
                else c = q;
                sum += c;
                word = word | ((c & 255) << (i * 8));
            }
            xq[b * 8 + w] = word;
        }
        xd[b * 2] = d;
        xd[b * 2 + 1] = (float)sum;
    }
}
