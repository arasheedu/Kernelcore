// ggml ROPE in NORM mode (adjacent pairs) or NEOX mode (pairs i and i + head_dim / 2, Qwen), out of
// place, with per-frequency factors (Llama 3; pass ones when the model has none) and per-token
// positions read from a tensor:
//   theta = pos[t] * base^(-2i/head_dim) / ff[i],  i = pair index within the head, t = token.
// x, y: [head_dim][n_head][n_tokens] contiguous. One thread per (token, pair index): the angle and
// its sine and cosine are computed once and applied to that pair of every head (as ggml caches
// them per position), so a token costs head_dim / 2 transcendental triples, not one per pair.
// dims = {pairs_per_head (head_dim / 2, a multiple of 8), n_head, n_tokens * pairs_per_head, m1, m2}:
// pair i of a head is at offsets i * m1 and i * m1 + m2 (NORM: 2, 1; NEOX: 1, pairs_per_head).

#ifndef PCUDA_ALIGNED8_DEFINED
#define PCUDA_ALIGNED8_DEFINED
// Promise that x is a non-negative multiple of 8 (identity). pcuda_mslgen uses it to keep
// x-based divisions uniform across the 8 lanes of a group, and checks it at run time.
__device__ int pcuda_aligned8(int x)
{
    return x;
}
#endif

extern "C" __global__ void rope_ggml(const float* x, float* y, const int* pos, const float* ff, const int* dims,
                                     float log2_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = pcuda_aligned8(dims[0]);
    const int nh = dims[1];
    const int count = dims[2];
    if (idx < count) {
        const int t = idx / pairs;
        const int i = idx - t * pairs;
        const float theta = (float)pos[t] * exp2f(-(float)i / (float)pairs * log2_base) / ff[i];
        const float c = cosf(theta);
        const float s = sinf(theta);
        const int m1 = dims[3];
        const int m2 = dims[4];
        for (int h = 0; h < nh; ++h) {
            const int a = 2 * (t * nh + h) * pairs + i * m1;
            const float x0 = x[a];
            const float x1 = x[a + m2];
            y[a] = x0 * c - x1 * s;
            y[a + m2] = x0 * s + x1 * c;
        }
    }
}
