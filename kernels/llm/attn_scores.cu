// scores[h][t] = scale * dot(q[h], K[t][h / group]) for every head h and cached position t.
// dims = {n_heads, ctx, head_dim, kv_stride, group}; kv_stride = n_kv_heads * head_dim.
// base: index of the first thread (0 on the GPU; the chunk start under pcuda chunking).
extern "C" __global__ void attn_scores(const float* q, const float* k, float* scores, const int* dims,
                                       int base, float scale)
{
    const int idx = base + blockIdx.x * blockDim.x + threadIdx.x;
    const int n_heads = dims[0];
    const int ctx = dims[1];
    const int hd = dims[2];
    const int kv_stride = dims[3];
    const int group = dims[4];
    if (idx < n_heads * ctx) {
        const int h = idx / ctx;
        const int t = idx % ctx;
        const int ko = t * kv_stride + (h / group) * hd;
        float s = 0.0f;
        for (int d = 0; d < hd; ++d) s += q[h * hd + d] * k[ko + d];
        scores[idx] = s * scale;
    }
}
