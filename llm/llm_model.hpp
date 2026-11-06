#pragma once

// A synthetic Llama-style decoder for benchmarking token generation (decode) the way llama.cpp
// does it: one token per step, K/V appended to the cache, attention over every position so far.
// Shapes follow real models; weights are random (4-bit Q4 or 8-bit Q8), so the output text
// would be meaningless, but the work and memory traffic per token are real.
//
// Simplifications (each small next to the weight traffic):
//   - the prompt's K/V cache entries are random instead of computed by a prefill pass;
//   - no embedding lookup (x starts from a fixed random vector each token);
//   - norms, attention and activations are f32; weight scales are f32 in a separate array
//     (Booth has no half type), so Q4 costs 5 bits per weight.

#include "llm_kernels.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// Q4X8: the Q4 bytes repacked with 8 rows interleaved (kernels/llm/matvec_q4x8.cu), so SIMD
// lanes and neighbouring GPU threads read adjacent bytes. Same size and results as Q4.
// Q4X8Q8: the same 4-bit weights with scales inside an 8-row interleaved stream, multiplied
// by 8-bit activations with integer dot products (AVX-VNNI on the simd path, __dp4a on CUDA).
enum class Weights { Q8, Q4, Q4X8, Q4X8Q8 };

struct Config {
    std::string name;
    int     hidden = 0;
    int     inter = 0;      // FFN width
    int     layers = 0;
    int     heads = 0;
    int     kv_heads = 0;
    int     head_dim = 0;
    int     vocab = 0;
    int     max_ctx = 0;    // KV cache capacity (prompt + generated tokens)
    float   eps = 1e-5f;
    float   rope_base = 500000.0f;
    Weights weights = Weights::Q4;

    int     kvStride() const { return kv_heads * head_dim; }
    int     group() const { return heads / kv_heads; }
    int     kStride() const { return padTo8(max_ctx); }  // transposed K cache row length
    AttnDims attnDims(int ctx) const { return {heads, ctx, head_dim, kvStride(), group(), padTo8(ctx), kStride()}; }
    int64_t weightParams() const;  // matrix elements (all layers + lm_head)
    int64_t weightBytes() const;   // quantized values + f32 scales: read once per token
};

Config llama32_3b(int max_ctx, Weights w);  // Llama 3.2 3B: `ollama run llama3.2`
Config llama32_1b(int max_ctx, Weights w);
Config tinyConfig(Weights w);               // small shapes for correctness checks

const char* weightsName(Weights w);

// Memory placement for one backend: pcuda, CUDA or plain host memory.
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* alloc(size_t bytes) = 0;
    virtual void  upload(void* dst, const void* src, size_t bytes) = 0;    // host -> this memory
    virtual void  download(void* dst, const void* src, size_t bytes) = 0;  // this memory -> host
};

class HostAllocator final : public Allocator {
public:
    ~HostAllocator() override;
    void* alloc(size_t bytes) override;
    void  upload(void* dst, const void* src, size_t bytes) override;
    void  download(void* dst, const void* src, size_t bytes) override;

private:
    std::vector<void*> blocks_;
};

class PcudaAllocator final : public Allocator {
public:
    ~PcudaAllocator() override;
    void* alloc(size_t bytes) override;
    void  upload(void* dst, const void* src, size_t bytes) override;
    void  download(void* dst, const void* src, size_t bytes) override;

private:
    std::vector<void*> blocks_;
};

struct Mat {
    int          rows = 0;
    int          k = 0;
    Weights      format = Weights::Q8;
    const void*  q = nullptr;     // int8 [rows][k] (Q8), nibbles [rows][k/2] (Q4) or interleaved (Q4X8)
    const float* s = nullptr;     // scales [rows][k/32]
    const int*   dims = nullptr;  // device int[2] {rows, k}
};

// q/k/v and gate/up projections are stored as one matrix each (rows concatenated), as many
// inference engines do: one launch (and one activation quantization) instead of three or two.
struct Layer {
    Mat          wqkv;  // [heads*head_dim + 2*kv_stride][hidden]: q rows, then k, then v
    Mat          wo;
    Mat          wgu;   // [2*inter][hidden]: gate rows, then up
    Mat          wd;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    float*       k_cache = nullptr;  // transposed: [kv_stride][kStride()]
    float*       v_cache = nullptr;  // [max_ctx][kv_stride]
};

struct Model {
    Config             cfg;
    std::vector<Layer> layers;
    const float*       out_norm = nullptr;
    Mat                lm_head;              // tied embeddings: [vocab][hidden]
    int*               attn_dims = nullptr;  // device int[7], rewritten for each token's context
};

struct Acts {
    float* x0 = nullptr;  // fixed start vector, copied into x at the start of each token
    float* x = nullptr;   // residual stream
    float *xn = nullptr, *scores = nullptr, *attn = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr;  // one buffer: k = q + heads*head_dim, v = k + kv_stride
    float *t = nullptr, *h = nullptr, *logits = nullptr;
    float *g = nullptr, *u = nullptr;  // one buffer: u = g + inter
    int*   xq = nullptr;  // 8-bit activation scratch for Q4X8Q8 (llm::scratchInts(max k / 32) ints)
};

// Builds the same deterministic model (same seed -> same bytes) into each allocator.
void buildModels(const Config& cfg, uint64_t seed, const std::vector<Allocator*>& allocs, std::vector<Model>& out);
Acts allocActs(const Config& cfg, Allocator& alloc, uint64_t seed);

// Kernel categories for the per-token breakdown.
enum class Op : int { MatvecAttn, MatvecFfn, MatvecLmHead, RmsNorm, Rope, Attention, SiluMul, Add, Count };
constexpr int kOpCount = static_cast<int>(Op::Count);
const char* opName(Op op);

class Profiler {
public:
    virtual ~Profiler() = default;
    virtual void begin(Op op) = 0;
    virtual void end(Op op) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    // Scratch for 8-bit activations (Q4X8Q8 weights): set once per token by decodeToken.
    virtual void setScratch(int* q) = 0;
    // Attention for the next launches covers positions [0, ctx): rewrites the device dims.
    virtual void setContext(int ctx) = 0;
    virtual void matvec(const Mat& w, const float* x, float* y) = 0;
    virtual void rmsNorm(const float* x, const float* w, float* y) = 0;  // one row of hidden
    virtual void addRmsNorm(float* x, const float* t, const float* w, float* y) = 0;  // x += t; y = norm(x)
    // Q4X8Q8 only: SwiGLU straight into the activation scratch, then a matvec that uses it.
    virtual void siluMulQuant(const float* g, const float* u, int n) = 0;
    virtual void matvecQuantized(const Mat& w, float* y) = 0;
    // Q4X8Q8 only: x += t, then RMS norm with weight w quantized straight into the scratch
    // (parallel: add_sumsq + norm_q8x4). The next matvecQuantized uses it.
    virtual void addRmsNormQuant(float* x, const float* t, const float* w) = 0;
    virtual void rope(float* x, int heads, int pos) = 0;
    virtual void appendKv(const float* k, const float* v, float* kt, float* vc, int pos) = 0;
    virtual void attnScores(const float* q, const float* k, float* scores) = 0;
    virtual void softmax(float* scores) = 0;
    virtual void attnValues(const float* p, const float* v, float* out) = 0;
    virtual void siluMul(const float* g, const float* u, float* out, int n) = 0;
    virtual void add(float* x, const float* t, int n) = 0;
    virtual void copy(float* dst, const float* src, int n) = 0;
    virtual void finish() = 0;  // wait until queued work is done
    // Bracket one token's launches (decodeToken calls them). A backend may record the launches
    // between them and run them as one batch (pcuda / CUDA graphs). `profiling`: per-op timing
    // is wanted, so run launches one by one.
    virtual void beginToken(bool profiling) { (void)profiling; }
    virtual void endToken() {}
};

// One decode step for the token at position `pos` (0-based): every layer (with K/V appended at
// `pos` and attention over pos + 1 positions), final norm and lm_head. Result in a.logits.
void decodeToken(Backend& b, const Model& m, const Acts& a, int pos, Profiler* prof = nullptr);

// Synchronous pcuda launches; the kernel code path follows pcudaSetKernelCodegen.
// workers == 0: one launch per kernel; workers > 0: matvec, attention, softmax and silu are
// chunked (pcudaLaunchKernelChunked with that many workers).
// graphs: run each token as one pcuda graph (pcudaStreamBeginCapture / pcudaGraphLaunch).
// two_rows: Q4X8Q8 matvecs with rows % 16 == 0 use matvec_q4x8_q8x2 (two rows per thread). Off by
// default: measured no faster on the CPU (memory is already saturated) or on CUDA.
std::unique_ptr<Backend> makePcudaBackend(const Model& m, unsigned workers, bool graphs = true, bool two_rows = false);

// Plain C++ on host memory (single thread), reading the weights of a pcuda Model directly
// (pcuda device memory is host memory).
std::unique_ptr<Backend> makeCppBackend(const Model& m);

} // namespace llm
