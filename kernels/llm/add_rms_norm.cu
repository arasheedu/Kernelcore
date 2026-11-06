// Residual add fused with the next RMS norm, for one row of n values (one thread):
//   x[i] = x[i] + t[i];  y = x / sqrt(mean(x^2) + eps) * w
// Same float operations, in the same order, as vector_add followed by rms_norm.
extern "C" __global__ void add_rms_norm(float* x, const float* t, const float* w, float* y, int n, float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < 1) {
        float ss = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float v = x[j] + t[j];
            x[j] = v;
            ss += v * v;
        }
        const float r = 1.0f / sqrtf(ss / (float)n + eps);
        for (int j = 0; j < n; ++j) y[j] = x[j] * r * w[j];
    }
}
