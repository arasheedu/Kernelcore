#pragma once

// Typed arguments and pcuda registration for the LLM decode kernels in kernels/llm/.
// The same .cu files are compiled by nvcc for the GPU side (gpu/cuda_llm.cu).

#include <pcuda/api.hpp>

#include <cstdint>

namespace llm {

// Attention shape shared by attn_scores and attn_values. Mirrors the device int[5] "dims".
struct AttnDims {
    int n_heads = 0;
    int ctx = 0;        // cached positions attended over
    int head_dim = 0;
    int kv_stride = 0;  // n_kv_heads * head_dim (one cache row)
    int group = 0;      // n_heads / n_kv_heads (grouped-query attention)
};

// y[rows] = W[rows][k] x[k], Q8 blocks of 32. dims -> device int[2] {rows, k}. k % 32 == 0.
struct MatvecQ8Args {
    const int8_t* q;
    const float*  scales;
    const float*  x;
    float*        y;
    const int*    dims;
    int           rows;
    int           k;
};

// y[rows] = W[rows][k] x[k] with 4-bit weights (ggml Q4_0 layout, float scale): qs is
// [rows][k/2] bytes, byte j of a block holds weights j (low nibble) and j+16 (high), each +8;
// scales [rows][k/32]. 5 bits per weight. dims -> device int[2] {rows, k}. k % 32 == 0.
struct MatvecQ4Args {
    const uint8_t* qs;
    const float*   scales;
    const float*   x;
    float*         y;
    const int*     dims;
    int            rows;
    int            k;
};

// matvec_q4x8 takes MatvecQ4Args with qs/scales in the 8-row interleaved layout (see
// kernels/llm/matvec_q4x8.cu and repackQ4x8). rows % 8 == 0.

// 8-bit activations for matvec_q4x8_q8 (kernels/llm/quantize_q8x4.cu): count blocks of 32.
struct QuantizeArgs {
    const float* x;   // [count * 32]
    int*         xq;  // [count][8] packed int8 words
    float*       xd;  // [count][2] {scale, sum of q}
    int          count;
};

// y = W x with 4-bit weights (8-row interleaved, scales inside the stream) and 8-bit activations.
// qw: rows * k * 5 / 8 bytes (see kernels/llm/matvec_q4x8_q8.cu and ref::repackQ4x8Q8).
struct MatvecQ4Q8Args {
    const int*   qw;
    const int*   xq;
    const float* xd;
    float*       y;
    const int*   dims;
    int          rows;
    int          k;
};

struct RmsNormArgs {  // one row of n per thread
    const float* x;
    const float* w;
    float*       y;
    int          n;
    int          rows;
    float        eps;
};

struct RopeArgs {  // in place; count = heads * pairs_per_head
    float* x;
    int    pairs_per_head;
    int    pos;
    int    count;
    float  log2_base;
};

struct AttnScoresArgs {  // scores[n_heads][ctx]; dims -> device int[5] matching d
    const float* q;
    const float* k;
    float*       scores;
    const int*   dims;
    AttnDims     d;
    float        scale;
};

struct SoftmaxArgs {  // in place, rows of n
    float* x;
    int    n;
    int    rows;
};

struct AttnValuesArgs {  // out[n_heads][head_dim]
    const float* p;
    const float* v;
    float*       out;
    const int*   dims;
    AttnDims     d;
};

struct SiluMulArgs {
    const float* g;
    const float* u;
    float*       out;
    int          count;
};

// Registers matvec_q8, matvec_q4, matvec_q4x8, quantize_q8x4, matvec_q4x8_q8, rms_norm, rope, attn_scores, softmax, attn_values and silu_mul.
// All but rope opt into chunked launches.
pcuda::Error registerKernels() noexcept;

} // namespace llm
