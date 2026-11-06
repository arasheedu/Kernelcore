// Test-only kernel: unbounded device recursion to exhaust the thread stack.
__device__ int deep(int n)
{
    if (n == 0) return 0;
    return deep(n - 1) + 1;
}
extern "C" __global__ void deep_recursion(int* out, int depth, int count)
{
    const int i = threadIdx.x;
    if (i < count) out[i] = deep(depth);
}