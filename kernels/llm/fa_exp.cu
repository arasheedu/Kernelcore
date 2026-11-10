// Softmax numerators of the attention scores, in place (see fa_scores):
// s[row * tp + t] = exp(s[row * tp + t] - m[row]) for t < T, one thread per entry (H * N * tp).
// dims as in fa_scores. base: index of the first thread.

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void fa_exp(float* s, const float* m, const int* dims, float base)
{
    const int idx = kc_aligned8((int)base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int tp = kc_aligned8(dims[4]);
    if (idx < dims[0] * dims[9] * tp) {
        const int row = idx / tp;
        const int t = idx - row * tp;
        if (t < dims[1]) s[idx] = exp2f((s[idx] - m[row]) * 1.44269504f);
    }
}
