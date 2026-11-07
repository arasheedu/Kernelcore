// First half of a parallel RMS norm when no residual add precedes it: per block of 32, the sum of
// x^2 into the row's scratch slot (see add_sumsq_f.cu for the layout, dims and base).

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void sumsq_part(const float* x, int* q, const int* dims, float base)
{
    const int idx = pcuda_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = pcuda_aligned8(dims[0]);
    const int n = dims[1];
    const int rows = dims[2];
    if (idx < nb * rows) {
        const int row = idx / nb;
        const int bl = idx - row * nb;
        const int o = row * n + bl * 32;
        float s = 0.0f;
        for (int i = 0; i < 32; ++i) s += x[o + i] * x[o + i];
        float * ss = (float *)(q + row * nb * 11 + nb * 10);
        ss[bl] = s;
    }
}
