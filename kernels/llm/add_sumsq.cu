// Residual add plus the first half of a parallel RMS norm, one thread per block of 32 values:
//   x[i] = x[i] + t[i];  ss[b] = sum of x[i]^2 over the block (in order)
// q is the 8-bit activation scratch of nb blocks: nb*8 packed words, nb*2 floats {scale, sum},
// then nb floats of partial sums ss (read by norm_q8x4).
// base: index of the first block (0 on the GPU; the chunk start under Kernelcore chunking).
extern "C" __global__ void add_sumsq(float* x, const float* t, int* q, int nb, int base)
{
    const int b = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) {
        float* ss = (float*)(q + nb * 10);
        float s = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = x[b * 32 + i] + t[b * 32 + i];
            x[b * 32 + i] = v;
            s += v * v;
        }
        ss[b] = s;
    }
}
