// In-place softmax over the first n values of each row (rows are `stride` floats apart),
// one thread per row.
extern "C" __global__ void softmax(float* x, int n, int stride, int rows)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float m = x[row * stride];
        for (int j = 1; j < n; ++j) m = fmaxf(m, x[row * stride + j]);
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = exp2f((x[row * stride + j] - m) * 1.44269504f);
            x[row * stride + j] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < n; ++j) x[row * stride + j] = x[row * stride + j] * inv;
    }
}
