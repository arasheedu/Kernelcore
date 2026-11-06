// SwiGLU: out[i] = silu(g[i]) * u[i], silu(g) = g / (1 + e^-g).
extern "C" __global__ void silu_mul(const float* g, const float* u, float* out, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float gi = g[i];
        out[i] = gi / (1.0f + exp2f(-gi * 1.44269504f)) * u[i];
    }
}
