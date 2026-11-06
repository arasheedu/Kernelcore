// Test kernel with float scalar parameters (xmm argument passing).
// Portable index (see kernels/vector_add.cu); also compiled by nvcc for pcuda_vs_cuda.
extern "C" __global__ void scale_add(const float* x, float* out, float scale, float bias, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = x[i] * scale + bias;
}
