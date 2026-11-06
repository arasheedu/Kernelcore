// Quantizes activations for matvec_q4x8_q8, like llama.cpp's Q8 activations: per block of 32
// values, d = max|x| / 127 and q = round(x / d) (half away from zero), clamped to [-127, 127].
//   xq [nblocks][8] ints: word w packs q[4w .. 4w+3], byte i = q[4w+i] (two's complement)
//   xd [nblocks][2] floats: {d, sum of the 32 q values (exact in float)}
// One thread per block. count = number of blocks.
// Every conditional assignment has an explicit else: Booth 0.6.1's --metal output drops the
// fall-through copy of an if without else (BOOTH_PIN F18), which the native paths are built from.
extern "C" __global__ void quantize_q8x4(const float* x, int* xq, float* xd, int count)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < count) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = x[b * 32 + i];
            amax = fmaxf(amax, fmaxf(v, 0.0f - v));
        }
        const float d = amax / 127.0f;
        float id;
        if (d > 0.0f) id = 1.0f / d;
        else id = 0.0f;
        int sum = 0;
        for (int w = 0; w < 8; ++w) {
            int word = 0;
            for (int i = 0; i < 4; ++i) {
                const float t = x[b * 32 + w * 4 + i] * id;
                int q;
                if (t >= 0.0f) q = (int)(t + 0.5f);
                else q = (int)(t - 0.5f);
                int c;
                if (q > 127) c = 127;
                else if (q < -127) c = -127;
                else c = q;
                sum += c;
                word = word | ((c & 255) << (i * 8));
            }
            xq[b * 8 + w] = word;
        }
        xd[b * 2] = d;
        xd[b * 2 + 1] = (float)sum;
    }
}
