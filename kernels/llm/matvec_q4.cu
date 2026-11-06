// y = W x for a Q4 weight matrix (ggml Q4_0 layout with a float scale): blocks of 32 weights,
// 16 bytes per block; byte j holds weight j (low nibble) and weight j + 16 (high nibble),
// each stored as value + 8. qs is [rows][k/2] bytes, scales [rows][k/32]. dims = {rows, k}.
// 5 bits per weight in total, the same bytes per token as Ollama's llama3.2 Q4_K_M file.
extern "C" __global__ void matvec_q4(const unsigned char* qs, const float* scales, const float* x,
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
            for (int j = 0; j < 16; ++j) {
                const int v = qs[row * (k / 2) + b * 16 + j];
                s += (float)((v & 15) - 8) * x[b * 32 + j];
                s += (float)((v >> 4) - 8) * x[b * 32 + j + 16];
            }
            acc += s * scales[row * nb + b];
        }
        y[row] = acc;
    }
}
