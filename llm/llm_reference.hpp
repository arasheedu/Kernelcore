#pragma once

// Plain C++ versions of kernels/llm/*.cu: same formulas, same loop order, float arithmetic.
// Used as the correctness oracle in tests and as the "C++ /O2" timing column.

#include "llm_kernels.hpp"

#include <cstdint>

namespace llm::ref {

void matvecQ8(const int8_t* q, const float* scales, const float* x, float* y, int rows, int k);
void matvecQ4(const uint8_t* qs, const float* scales, const float* x, float* y, int rows, int k);
// 8-row interleaved Q4 (kernels/llm/matvec_q4x8.cu), 4 partial sums per block. rows % 8 == 0.
void matvecQ4x8(const uint8_t* qs, const float* scales, const float* x, float* y, int rows, int k);
// Row-major Q4 bytes/scales -> 8-row interleaved layout.
void repackQ4x8(const uint8_t* qs, const float* scales, uint8_t* qs8, float* scales8, int rows, int k);
// 8-bit activations (kernels/llm/quantize_q8x4.cu) and the dot-product matvec that uses them.
void quantizeQ8x4(const float* x, int* xq, float* xd, int count);
void matvecQ4x8Q8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k);
// Row-major Q4 bytes/scales -> the matvec_q4x8_q8 weight stream (rows * k * 5 / 8 bytes).
void repackQ4x8Q8(const uint8_t* qs, const float* scales, int* qw, int rows, int k);
void rmsNorm(const float* x, const float* w, float* y, int n, int rows, float eps);
// Fused kernels: the same results as the unfused sequences.
void addRmsNorm(float* x, const float* t, const float* w, float* y, int n, float eps);  // add, then rmsNorm
void siluMulQ8x4(const float* g, const float* u, int* xq, float* xd, int count);          // siluMul, then quantizeQ8x4
// Parallel RMS norm + quantization (add_sumsq, norm_q8x4); q is the scratch of llm::scratchInts(nb).
void addSumsq(float* x, const float* t, int* q, int nb);
void normQ8x4(const float* x, const float* w, int* q, int nb, float eps);

// GGUF K-quants. Streams: rows in groups of 8, super-blocks interleaved by 32-bit word
// (see kernels/llm/matvec_q4k_q8.cu and matvec_q6k_q8.cu); fp16 scales are stored as f32.
float f16ToF32(uint16_t h);
void  dequantRowQ4K(const uint8_t* src, int k, float* out);  // ggml dequantize_row_q4_K
void  dequantRowQ6K(const uint8_t* src, int k, float* out);  // ggml dequantize_row_q6_K
void  repackQ4K(const uint8_t* src, int rows, int k, int* qw);  // rows * k / 256 * 37 ints
void  repackQ6K(const uint8_t* src, int rows, int k, int* qw);  // rows * k / 256 * 53 ints
void  matvecQ4KQ8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k);
void  matvecQ6KQ8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k);
void  ropeFF(float* x, const float* ff, int pairs_per_head, int pos, int count, float log2_base);
void rope(float* x, int pairs_per_head, int pos, int count, float log2_base);
// Layouts as in AttnDims: kt[kv_stride][k_stride] (K transposed), v[ctx][kv_stride],
// scores/p[n_heads][ctx_pad] (entries t >= ctx untouched).
void appendKv(const float* k, const float* v, float* kt, float* vc, int n, int k_stride, int pos);
void attnScores(const float* q, const float* kt, float* scores, const AttnDims& d, float scale);
void softmax(float* x, int n, int stride, int rows);
void attnValues(const float* p, const float* v, float* out, const AttnDims& d);
void siluMul(const float* g, const float* u, float* out, int count);
void add(float* x, const float* t, int n);

} // namespace llm::ref
