// Row maxima of the attention scores (see fa_scores): m[row] = max over t < T of s[row * tp + t],
// one thread per row (H * N rows). dims as in fa_scores. base: index of the first thread.

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void fa_max(const float* s, float* m, const int* dims, float base)
{
    const int row = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int T = dims[1];
    const int tp = dims[4];
    if (row < dims[0] * dims[9]) {
        const int r = row * tp;
        float mx = s[r];
        for (int t = 1; t < T; ++t) mx = fmaxf(mx, s[r + t]);
        m[row] = mx;
    }
}
