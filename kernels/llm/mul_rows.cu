// y[i] = x[i] * w[i % n]: a row vector w[n] broadcast over count / n rows (ggml MUL by a norm
// weight). n is a multiple of 8 (pcuda_aligned8), so i % n stays contiguous across simd lanes.

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void mul_rows(const float* x, const float* w, float* y, int n, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n8 = pcuda_aligned8(n);
    if (i < count) {
        y[i] = x[i] * w[i % n8];
    }
}
