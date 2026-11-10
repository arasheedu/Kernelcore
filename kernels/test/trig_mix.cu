// Calls the four libm externals Booth emits (sinf, cosf, exp2f, log2f).
// Portable index (see kernels/vector_add.cu); also compiled by nvcc for kc_vs_cuda.
extern "C" __global__ void trig_mix(const float* x, float* out, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = sinf(x[i]) + cosf(x[i]) + exp2f(x[i]) + log2f(x[i] + 2.0f);
}
