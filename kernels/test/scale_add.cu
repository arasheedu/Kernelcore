// Test kernel with float scalar parameters (xmm argument passing).
extern "C" __global__ void scale_add(const float* x, float* out, float scale, float bias, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i] = x[i] * scale + bias;
}