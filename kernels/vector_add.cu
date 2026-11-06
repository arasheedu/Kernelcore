extern "C" __global__ void vector_add(const float* a, const float* b,
                                      float* out, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}