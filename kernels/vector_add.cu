// Portable index: on a GPU this is the global thread index. Under Booth --cpu,
// blockIdx.x == 0 and blockDim.x == nthreads, so it reduces to threadIdx.x.
extern "C" __global__ void vector_add(const float* a, const float* b,
                                      float* out, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}
