#pragma once

// Typed arguments and Kernelcore registration for the LLM decode kernels in kernels/llm/.
// The same .cu files are compiled by nvcc for the GPU side (gpu/cuda_llm.cu).

#include <kernelcore/api.hpp>

#include <cstdint>

namespace llm {

// Attention shape shared by attn_scores and attn_values. Mirrors the device int[7] "dims".
// Layouts: K transposed kt[kv_stride][k_stride]; V v[ctx][kv_stride]; scores p[n_heads][ctx_pad].
struct AttnDims {
    int n_heads = 0;
    int ctx = 0;        // cached positions attended over
    int head_dim = 0;   // multiple of 8
    int kv_stride = 0;  // n_kv_heads * head_dim (one V cache row)
    int group = 0;      // n_heads / n_kv_heads (grouped-query attention)
    int ctx_pad = 0;    // scores row length: ctx rounded up to a multiple of 8
    int k_stride = 0;   // transposed K row length (cache capacity), >= ctx
};
inline int padTo8(int n) { return (n + 7) / 8 * 8; }

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

// Fused residual add + RMS norm of one row (kernels/llm/add_rms_norm.cu): x += t; y = norm(x) * w.
struct AddRmsNormArgs {
    float*       x;
    const float* t;
    const float* w;
    float*       y;
    int          n;
    float        eps;
};

// 8-bit activation scratch for nb blocks of 32 values, one int buffer:
//   [nb * 8] packed words | [nb * 2] floats {scale, sum} | [nb] floats (RMS-norm partial sums)
inline int    scratchInts(int nb) { return nb * 11; }
inline float* scratchScales(int* q, int nb) { return reinterpret_cast<float*>(q + nb * 8); }
// 8-bit activations with one scale per 256 values (quantize_cols_k, matmul_*_q8kx*), per column:
//   [nsb * 64] packed words | [nsb] float scales | [nsb * 16] int sums of 16 values |
//   [nsb * 4] the sums of 32 as bytes (high parts B >> 6, then low parts B & 63)
inline int    scratchKInts(int nsb) { return nsb * 85; }

// Parallel residual add + RMS norm + 8-bit quantization, in two launches (both chunkable):
//   add_sumsq:  x += t; per-block sums of squares into the scratch
//   norm_q8x4:  y = x / sqrt(mean(x^2) + eps) * w, quantized into the scratch (y not stored)
struct AddSumsqArgs {
    float*       x;  // [nb * 32]
    const float* t;
    int*         q;  // scratchInts(nb)
    int          nb;
};
struct NormQ8Args {
    const float* x;
    const float* w;
    int*         q;
    int          nb;
    float        eps;
};

// Fused SwiGLU + 8-bit quantization (kernels/llm/silu_mul_q8x4.cu): count blocks of 32.
struct SiluMulQ8Args {
    const float* g;
    const float* u;
    int*         xq;
    float*       xd;
    int          count;
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

// Rope with per-frequency factors ff[pairs_per_head] (Llama 3 rope scaling, GGUF rope_freqs).
struct RopeFFArgs {  // in place; count = heads * pairs_per_head
    float*       x;
    const float* ff;
    int          pairs_per_head;
    int          pos;
    int          count;
    float        log2_base;
};

// y[i] = x[i] * w[i % n] (row vector broadcast), n % 8 == 0. Not chunked.
struct MulRowsArgs {
    const float* x;
    const float* w;
    float*       y;
    int          n;
    int          count;
};

// ggml ROPE, NORM mode, out of place; pos[n_tokens] per-token positions, ff[head_dim / 2]
// frequency factors (ones when the model has none); dims -> device int[3]
// {head_dim / 2, n_head, n_tokens * head_dim / 2}; head_dim % 16 == 0. Not chunked.
struct RopeGgmlArgs {
    const float* x;
    float*       y;
    const int*   pos;
    const float* ff;
    const int*   dims;
    int          head_dim;
    int          n_head;
    int          n_tokens;
    float        log2_base;
    bool         neox = false;  // NEOX pairs (i, i + head_dim / 2); dims -> int[5] (see rope_ggml.cu)
};

// Parallel residual add + RMS norm + weight + 8-bit quantization for ggml graphs (kernels
// add_sumsq_f / sumsq_part, then norm_mul_q8_f), and SwiGLU + quantization (silu_mul_q8x4f).
// Rows of n values (n % 256 == 0); q: one scratch slot of llm::scratchInts(n / 32) ints per row.
// dims -> device int[3] {n / 32, n, rows} (silu: int[2] {n, cols}). All chunkable.
struct AddSumsqFArgs {  // out = a + b, partial sums of out^2 into q
    const float* a;
    const float* b;
    float*       out;
    int*         q;
    const int*   dims;
    int          n;
    int          rows;
};
struct SumsqPartArgs {  // partial sums of x^2 into q
    const float* x;
    int*         q;
    const int*   dims;
    int          n;
    int          rows;
};
struct NormMulQ8Args {  // y = x / rms * w (rms from q's partial sums), quantized into q
    const float* x;
    const float* w;
    float*       y;
    int*         q;
    const int*   dims;
    int          n;
    int          rows;
    float        eps;
};
struct SiluMulQ8FArgs {  // y = silu(g) * u, quantized into q
    const float* g;
    const float* u;
    float*       y;
    int*         q;
    const int*   dims;
    int          n;
    int          cols;
};

// For batches, norm_mul_q8k_f and silu_mul_q8kf take the same arguments and quantize per 256 values
// into quantize_cols_k slots (scratchKInts(n / 256) ints per row/column). norm_mul_q8k_f: q holds
// rows * scratchKInts(n / 256) ints of those slots, then rows * scratchInts(n / 32) ints whose partial
// sums add_sumsq_f / sumsq_part fill (pass them q + rows * scratchKInts(n / 256));
// dims -> int[4] {n / 256, n, rows, rows * scratchKInts(n / 256)}.

struct AttnScoresArgs {  // scores[n_heads][ctx_pad]; dims -> device int[7] matching d
    const float* q;
    const float* k;
    float*       scores;
    const int*   dims;
    AttnDims     d;
    float        scale;
};

struct SoftmaxArgs {  // in place: the first n values of each row; rows are `stride` floats apart
    float* x;
    int    n;
    int    stride;
    int    rows;
};

struct KvAppendArgs {  // writes position d.ctx - 1 of kt[kv_stride][k_stride] (K transposed), vc[ctx][kv_stride]
    const float* k;
    const float* v;
    float*       kt;
    float*       vc;
    const int*   dims;  // device int[7] matching d
    AttnDims     d;
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

// Y = W X for GGUF K-quant weights and ncols 8-bit activation columns (kernels/llm/
// matmul_q4k_q8x4.cu, matmul_q6k_q8x4.cu): each weight word is read once per 4 columns. Column c's
// activations are the scratch slot xq + c * scratchInts(k / 32); its output is y + c * rows.
// Bit-identical to matvec_q4k_q8 / matvec_q6k_q8 per column. matmul_q4k_q8kx4/x8 and
// matmul_q6k_q8kx4/x8 take the same arguments with quantize_cols_k slots (scratchKInts(k / 256)).
// dims -> device int[4]
// {rows, k, ncols, row_block}: threads run in blocks of row_block 8-row groups, tile by tile.
struct MatmulKQ8Args {
    const int* qw;
    const int* xq;
    float*     y;
    const int* dims;
    int        rows;
    int        k;
    int        ncols;
    int        row_block = 8;
};

// 8-bit quantization of ncols columns of k floats (x + c * k) into scratch slots
// q + c * scratchInts(k / 32), as quantize_q8x4 does (kernels/llm/quantize_cols.cu).
// dims -> device int[2] {k / 32, ncols}. quantize_cols_k takes the same arguments: slots of
// scratchKInts(k / 256), one scale per 256 values, dims {k / 256, ncols}.
struct QuantizeColsArgs {
    const float* x;
    int*         q;
    const int*   dims;
    int          k;
    int          ncols;
};

// Y = W X for fp16 weights (kernels/llm/matmul_f16.cu): W repacked by llm::ref::repackF16 (rows in
// groups of 8, interleaved by 32-bit word), x: ncols columns of k floats, y: ncols columns of rows.
// rows % 8 == 0, k even. dims -> device int[3] {rows, k, ncols}. Chunkable (thread start as a float).
struct MatmulF16Args {
    const int*   qw;
    const float* x;
    float*       y;
    const int*   dims;
    int          rows;
    int          k;
    int          ncols;
};

// ggml SET_ROWS into an fp16 tensor: dst row idx[r] (int64) = fp16(src row r), rows of 2 * pairs
// values (pairs % 8 == 0); rows outside [0, dst_rows) are skipped. Strides in floats / 32-bit words.
// dims -> device int[5] {pairs, rows, src_row, dst_row, dst_rows}.
struct SetRowsF16Args {
    const float* src;
    const int*   idx;
    int*         dst;
    const int*   dims;
    int          pairs;
    int          rows;
    int          src_row;
    int          dst_row;
    int          dst_rows;
};

// ggml FLASH_ATTN_EXT with an fp16 KV cache, in four kernels: fa_scores, fa_max, fa_exp, fa_values
// (kernels/llm/fa_scores.cu has the layouts). Scores s: H * N rows of tp floats.
// dims -> device int[15], the fields in this order.
struct FaDims {
    int H = 0;         // query heads
    int T = 0;         // KV positions
    int D = 0;         // head size (K and V), a multiple of 8
    int group = 0;     // H / H_kv
    int tp = 0;        // score row length: T rounded up to a multiple of 8
    int k_row = 0;     // K strides in 32-bit words (two fp16)
    int k_head = 0;
    int v_row = 0;     // V strides in words
    int v_head = 0;
    int N = 0;         // queries (tokens)
    int q_tok = 0;     // Q strides in floats
    int q_head = 0;
    int m_row = 0;     // mask row stride in words
    int has_mask = 0;
    int H_kv = 0;      // KV heads
};
constexpr int kFaDims = 15;
inline void faDimsInts(const FaDims& d, int* out) {
    const int v[kFaDims] = {d.H,    d.T,     d.D, d.group, d.tp,    d.k_row,    d.k_head, d.v_row,
                            d.v_head, d.N, d.q_tok, d.q_head, d.m_row, d.has_mask, d.H_kv};
    for (int i = 0; i < kFaDims; ++i) out[i] = v[i];
}
struct FaScoresArgs {
    const float* q;
    const int*   k;
    const int*   mask;  // any valid allocation when !d.has_mask
    float*       s;
    const int*   dims;
    FaDims       d;
    float        scale;
};
struct FaMaxArgs {  // m[H * N]
    const float* s;
    float*       m;
    const int*   dims;
    FaDims       d;
};
struct FaExpArgs {
    float*       s;
    const float* m;
    const int*   dims;
    FaDims       d;
};
struct FaValuesArgs {  // out[N][H][D]
    const float* p;
    const int*   v;
    float*       out;
    const int*   dims;
    FaDims       d;
};

// Registers matvec_q8, matvec_q4, matvec_q4x8, quantize_q8x4, quantize_cols, matvec_q4x8_q8, matvec_q4x8_q8x2, matvec_q4k_q8, matvec_q6k_q8, matmul_q4k_q8x4, matmul_q6k_q8x4, rope_ff, add_rms_norm, silu_mul_q8x4, add_sumsq, norm_q8x4, rms_norm, rope, kv_append, attn_scores, softmax, attn_values, silu_mul, matmul_f16, set_rows_f16 and fa_scores / fa_max / fa_exp / fa_values.
// All but add_rms_norm, rope and kv_append opt into chunked launches.
kc::Error registerKernels() noexcept;

} // namespace llm
