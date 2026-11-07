// Rotary position embedding with per-frequency factors (Llama 3's rope scaling, stored in the
// GGUF as rope_freqs.weight), adjacent-pair layout, in place. One thread per pair:
//   theta = pos * base^(-2i/head_dim) / ff[i],  i = pair index within the head.
// Otherwise the same as rope.cu.
extern "C" __global__ void rope_ff(float* x, const float* ff, int pairs_per_head, int pos, int count,
                                   float log2_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const int i = idx % pairs_per_head;
        const float theta = (float)pos * exp2f(-(float)i / (float)pairs_per_head * log2_base) / ff[i];
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = x[2 * idx];
        const float x1 = x[2 * idx + 1];
        x[2 * idx] = x0 * c - x1 * s;
        x[2 * idx + 1] = x0 * s + x1 * c;
    }
}
