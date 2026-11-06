// out[h][d] = sum over t of p[h][t] * V[t][h / group][d], one thread per (head, dim).
// dims = {n_heads, ctx, head_dim, kv_stride, group}. base as in attn_scores.
extern "C" __global__ void attn_values(const float* p, const float* v, float* out, const int* dims, int base)
{
    const int idx = base + blockIdx.x * blockDim.x + threadIdx.x;
    const int n_heads = dims[0];
    const int ctx = dims[1];
    const int hd = dims[2];
    const int kv_stride = dims[3];
    const int group = dims[4];
    if (idx < n_heads * hd) {
        const int h = idx / hd;
        const int d = idx % hd;
        const int vo = (h / group) * hd + d;
        float s = 0.0f;
        for (int t = 0; t < ctx; ++t) s += p[h * ctx + t] * v[t * kv_stride + vo];
        out[idx] = s;
    }
}
