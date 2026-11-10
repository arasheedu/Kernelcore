// out[h][d] = sum over t of p[h][t] * V[t][h / group][d], one thread per (head, dim).
// p rows are ctx_pad long (see attn_scores). dims = {n_heads, ctx, head_dim, kv_stride, group,
// ctx_pad, k_stride}; head_dim is a multiple of 8. base as in attn_scores.

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void attn_values(const float* p, const float* v, float* out, const int* dims, int base)
{
    const int idx = kc_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int n_heads = dims[0];
    const int ctx = dims[1];
    const int hd = kc_aligned8(dims[2]);
    const int kv_stride = dims[3];
    const int group = dims[4];
    const int cp = dims[5];
    if (idx < n_heads * hd) {
        const int h = idx / hd;
        const int d = idx - h * hd;
        const int vo = (h / group) * hd + d;
        float s = 0.0f;
        for (int t = 0; t < ctx; ++t) s += p[h * cp + t] * v[t * kv_stride + vo];
        out[idx] = s;
    }
}
