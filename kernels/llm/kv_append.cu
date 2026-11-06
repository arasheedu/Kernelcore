// Appends one token's K and V at position pos = ctx - 1 (the token being decoded).
// K is stored transposed, kt[j * k_stride + pos], for attn_scores; V row-major, v[pos * n + j].
// dims: the attention dims {n_heads, ctx, head_dim, kv_stride (= n), group, ctx_pad, k_stride}.
extern "C" __global__ void kv_append(const float* k, const float* v, float* kt, float* vc, const int* dims)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos = dims[1] - 1;
    const int n = dims[3];
    const int ks = dims[6];
    if (j < n) {
        kt[j * ks + pos] = k[j];
        vc[pos * n + j] = v[j];
    }
}
