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
void rmsNorm(const float* x, const float* w, float* y, int n, int rows, float eps);
void rope(float* x, int pairs_per_head, int pos, int count, float log2_base);
void attnScores(const float* q, const float* k, float* scores, const AttnDims& d, float scale);
void softmax(float* x, int n, int rows);
void attnValues(const float* p, const float* v, float* out, const AttnDims& d);
void siluMul(const float* g, const float* u, float* out, int count);
void add(float* x, const float* t, int n);

} // namespace llm::ref
