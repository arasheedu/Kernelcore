// Second half of a parallel RMS norm, times the weight row, with both outputs: f32 y and its 8-bit
// quantization for the matmuls that read it. One thread per (row, block of 32):
//   r = 1 / sqrt(sum of the row's partial sums / n + eps)   (every lane adds them in the same order)
//   y = x * r * w;  the block of y quantized exactly as quantize_q8x4 does it.
// Partial sums come from add_sumsq_f / sumsq_part (same q layout, dims and base).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void norm_mul_q8_f(const float* x, const float* w, float* y, int* q, const int* dims, float eps,
                                         float base)
{
    const int idx = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = kc_aligned8(dims[0]);
    const int n = dims[1];
    const int rows = dims[2];
    if (idx < nb * rows) {
        const int row = idx / nb;
        const int bl = idx - row * nb;
        int * qr = q + row * nb * 11;
        float * xd = (float *)(qr + nb * 8);
        const float * ss = (const float *)(qr + nb * 10);
        float tot = 0.0f;
        for (int j = 0; j < nb; ++j) tot += ss[j];
        const float r = 1.0f / sqrtf(tot / (float)n + eps);
        const int o = row * n + bl * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = x[o + i] * r * w[bl * 32 + i];
            y[o + i] = v;
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
                const float t = y[o + k * 4 + i] * id;
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
            qr[bl * 8 + k] = word;
        }
        xd[bl * 2] = d;
        xd[bl * 2 + 1] = (float)sum;
    }
}
