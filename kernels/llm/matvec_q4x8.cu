// y = W x for 4-bit weights stored 8 rows interleaved ("x8"): for each group of 8 rows and each
// block of 32 weights, byte j (j = 0..15) of all 8 rows is stored side by side. Neighbouring
// threads (rows) then read neighbouring bytes: one contiguous load for 8 SIMD lanes on the CPU,
// coalesced loads on a GPU.
//   qs     [rows/8][k/32][16][8] bytes: weight j (low nibble) and j+16 (high nibble), each +8
//   scales [rows/8][k/32][8]
// rows must be a multiple of 8. dims = {rows, k}.
// Each block is summed in 4 independent partial sums (combined in a fixed order), so a CPU core
// is not stalled on one chain of dependent additions. Every backend runs this same order.
extern "C" __global__ void matvec_q4x8(const unsigned char* qs, const float* scales, const float* x,
                                       float* y, const int* dims)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = dims[0];
    const int k = dims[1];
    if (row < rows) {
        const int nb = k / 32;
        const int g = row / 8;
        const int l = row - g * 8;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const int blk = g * nb + b;
            const int base = blk * 128 + l;
            const int xb = b * 32;
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            for (int j = 0; j < 16; j += 2) {
                const int v0 = qs[base + j * 8];
                const int v1 = qs[base + j * 8 + 8];
                s0 += (float)((v0 & 15) - 8) * x[xb + j];
                s1 += (float)((v0 >> 4) - 8) * x[xb + j + 16];
                s2 += (float)((v1 & 15) - 8) * x[xb + j + 1];
                s3 += (float)((v1 >> 4) - 8) * x[xb + j + 17];
            }
            acc += ((s0 + s2) + (s1 + s3)) * scales[blk * 8 + l];
        }
        y[row] = acc;
    }
}
