extern "C" __global__ void trig_mix(const float* x, float* out, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i] = sinf(x[i]) + cosf(x[i]) + exp2f(x[i]) + log2f(x[i] + 2.0f);
}