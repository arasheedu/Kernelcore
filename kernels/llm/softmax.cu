// In-place softmax over each row of n values, one thread per row.
extern "C" __global__ void softmax(float* x, int n, int rows)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float m = x[row * n];
        for (int j = 1; j < n; ++j) m = fmaxf(m, x[row * n + j]);
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = exp2f((x[row * n + j] - m) * 1.44269504f);
            x[row * n + j] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < n; ++j) x[row * n + j] = x[row * n + j] * inv;
    }
}
