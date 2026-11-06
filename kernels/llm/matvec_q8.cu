// y = W x for a Q8 weight matrix: int8 values in blocks of 32, one float scale per block.
// W is row-major [rows][k] (the ggml layout); scales are [rows][k/32]. dims = {rows, k}.
// One thread per output row. Booth: at most 5 pointer/int parameters, so sizes come in dims.
extern "C" __global__ void matvec_q8(const signed char* q, const float* scales, const float* x,
                                     float* y, const int* dims)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = dims[0];
    const int k = dims[1];
    if (row < rows) {
        const int nb = k / 32;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            float s = 0.0f;
            for (int j = 0; j < 32; ++j)
                s += (float)q[row * k + b * 32 + j] * x[b * 32 + j];
            acc += s * scales[row * nb + b];
        }
        y[row] = acc;
    }
}
