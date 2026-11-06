// Test-only kernel: integer division; divisor 0 raises a divide-by-zero fault.
extern "C" __global__ void int_div(const int* a, int* out, int divisor, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i] = a[i] / divisor;
}