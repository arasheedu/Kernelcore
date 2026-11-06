// Second half of a parallel RMS norm, fused with 8-bit quantization, one thread per block of 32:
//   r = 1 / sqrt(sum(ss) / n + eps)   (every thread adds the nb partial sums in the same order)
//   h = x * r * w, then exactly what quantize_q8x4 does to h (h itself is never stored).
// q: the scratch written by add_sumsq (nb*8 words, nb*2 floats {scale, sum}, nb floats ss).
// Every conditional assignment has an explicit else (BOOTH_PIN F18). base as in add_sumsq.
extern "C" __global__ void norm_q8x4(const float* x, const float* w, int* q, int nb, int base, float eps)
{
    const int b = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) {
        float* xd = (float*)(q + nb * 8);
        const float* ss = (const float*)(q + nb * 10);
        float tot = 0.0f;
        for (int j = 0; j < nb; ++j) tot += ss[j];
        const float r = 1.0f / sqrtf(tot / (float)(nb * 32) + eps);
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = x[b * 32 + i] * r * w[b * 32 + i];
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int sum = 0;
        for (int k = 0; k < 8; ++k) {
            int word = 0;
            for (int i = 0; i < 4; ++i) {
                const float v = x[b * 32 + k * 4 + i] * r * w[b * 32 + k * 4 + i];
                const float t = v * id;
                int qi;
                if (t >= 0.0f) qi = (int)(t + 0.5f);
                else qi = (int)(t - 0.5f);
                int c;
                if (qi > 127) c = 127;
                else if (qi < -127) c = -127;
                else c = qi;
                sum += c;
                word = word | ((c & 255) << (i * 8));
            }
            q[b * 8 + k] = word;
        }
        xd[b * 2] = d;
        xd[b * 2 + 1] = (float)sum;
    }
}
