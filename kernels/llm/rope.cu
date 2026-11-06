// Rotary position embedding (Llama "normal" layout: adjacent pairs), in place.
// One thread per pair; pairs_per_head = head_dim / 2. theta = pos * base^(-2i/head_dim),
// written with exp2f/log2f because Booth only links sinf, cosf, exp2f and log2f.
extern "C" __global__ void rope(float* x, int pairs_per_head, int pos, int count, float log2_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const int i = idx % pairs_per_head;
        const float theta = (float)pos * exp2f(-(float)i / (float)pairs_per_head * log2_base);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = x[2 * idx];
        const float x1 = x[2 * idx + 1];
        x[2 * idx] = x0 * c - x1 * s;
        x[2 * idx + 1] = x0 * s + x1 * c;
    }
}
