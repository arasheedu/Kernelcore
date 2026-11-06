// Test-only kernel: writes one full buffer length past the end of `out`.
extern "C" __global__ void oob_write(float* out, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i + count] = 1.0f;
}