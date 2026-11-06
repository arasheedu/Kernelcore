// y = x / sqrt(mean(x^2) + eps) * w, one thread per row of n values.
extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float ss = 0.0f;
        for (int j = 0; j < n; ++j) ss += x[row * n + j] * x[row * n + j];
        const float r = 1.0f / sqrtf(ss / (float)n + eps);
        for (int j = 0; j < n; ++j) y[row * n + j] = x[row * n + j] * r * w[j];
    }
}
