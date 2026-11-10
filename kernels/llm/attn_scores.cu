// scores[h][t] = scale * dot(q[h], K[h / group][.][t]) for every head h and cached position t.
// K is stored transposed: kt[(kv_head * head_dim + d) * k_stride + t], so threads with
// consecutive positions read consecutive floats.
// Scores rows are ctx_pad long (a multiple of 8); entries t >= ctx are not written.
// dims = {n_heads, ctx, head_dim, kv_stride, group, ctx_pad, k_stride}.
// base: index of the first thread (0 on the GPU; the chunk start under Kernelcore chunking).

#ifndef KC_ALIGNED8_DEFINED
#define KC_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). kc_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int kc_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void attn_scores(const float* q, const float* kt, float* scores, const int* dims,
                                       int base, float scale)
{
    const int idx = kc_aligned8(base) + blockIdx.x * blockDim.x + threadIdx.x;
    const int n_heads = dims[0];
    const int ctx = dims[1];
    const int hd = dims[2];
    const int group = dims[4];
    const int cp = kc_aligned8(dims[5]);
    const int ks = dims[6];
    if (idx < n_heads * cp) {
        const int h = idx / cp;
        const int t = idx - h * cp;
        if (t < ctx) {
            const int kb = (h / group) * hd;
            float s = 0.0f;
            for (int d = 0; d < hd; ++d) s += q[h * hd + d] * kt[(kb + d) * ks + t];
            scores[idx] = s * scale;
        }
    }
}
