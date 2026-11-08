// ggml backend for pcuda: MUL_MAT with Q4_K / Q6_K weights runs as CUDA C kernels on the CPU cores
// (CudaOnCPUCores: kernels/llm/matvec_q4k_q8.cu, matvec_q6k_q8.cu through the pcuda runtime).
//
// - Device: an ACCEL device, so llama.cpp offers its buffer type first for every weight. Only
//   MUL_MAT with a Q4_K / Q6_K weight (2-D, k % 256 == 0, rows % 8 == 0) is supported; every other
//   weight and op stays with the CPU backend.
// - Buffer type: pcuda memory (pcudaMalloc). set_tensor repacks Q4_K / Q6_K weights into the
//   kernels' 8-row interleaved stream (fp16 scales widened to f32: +3% / +1% bytes); get_tensor
//   undoes it. It is not a host buffer type (its weights are not in ggml layout), so the CPU
//   backend never reads them; outputs are copied to CPU memory by the scheduler.
// - Compute: each activation column (f32, host memory) is quantized to 8-bit blocks of 32 on the
//   host, then the matvec kernel runs chunked over pcuda's worker pool. All MUL_MATs of one
//   graph_compute call are recorded as one pcuda graph and run as one job.

#include "ggml-pcuda.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <pcuda/api.hpp>
#include <pcuda/kernels.hpp>

#include "llm_kernels.hpp"
#include "llm_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---- pcuda runtime (process-wide) ----------------------------------------------

struct PcudaRuntime {
    bool     ok = false;
    unsigned workers = 1;          // pool threads + caller: compute-bound batched matmuls use them all
    unsigned fast_workers = 1;     // the fastest cores' logical processors: every other (bandwidth-bound) launch
    bool     graphs = true;
    bool     disabled = false;     // GGML_PCUDA_DISABLE=1: claim nothing (CPU-only comparisons)
    bool     matmul_only = false;  // GGML_PCUDA_MATMUL_ONLY=1: only MUL_MAT (the first prototype)
    bool     matvec_cols = false;  // GGML_PCUDA_MATVEC_COLS=1: one matvec per column for batches (no matmul kernel)
    int      row_block = 8;        // GGML_PCUDA_ROW_BLOCK: 8-row groups per matmul block (cache blocking)
    int      tile = 8;             // GGML_PCUDA_TILE=4 or 8: columns per matmul thread (batches of <= 4 use 4)
    bool     q8k = true;           // GGML_PCUDA_Q8K=0: batches use per-32 activation scales (bit-identical to the matvec)
    int      matmul_chunks = 4;    // GGML_PCUDA_MATMUL_CHUNKS: chunk factor for batched matmuls (fast cores take more chunks)
};

PcudaRuntime & runtime() {
    static PcudaRuntime rt = [] {
        PcudaRuntime r;
        if (pcudaInit() != pcuda::Error::Success) {
            GGML_LOG_ERROR("pcuda: pcudaInit failed: %s\n", pcudaGetLastErrorDetail());
            return r;
        }
        if (llm::registerKernels() != pcuda::Error::Success) {
            GGML_LOG_ERROR("pcuda: registering kernels failed: %s\n", pcudaGetLastErrorDetail());
            return r;
        }
        pcuda::DeviceProperties p{};
        pcudaGetDeviceProperties(&p, 0);
        r.workers = p.worker_count + 1;
        r.fast_workers = std::max(1u, p.performance_workers);
        const char * g = std::getenv("GGML_PCUDA_GRAPHS");
        r.graphs = !(g && std::strcmp(g, "0") == 0);
        const char * off = std::getenv("GGML_PCUDA_DISABLE");
        r.disabled = off && std::strcmp(off, "1") == 0;
        const char * mm = std::getenv("GGML_PCUDA_MATMUL_ONLY");
        r.matmul_only = mm && std::strcmp(mm, "1") == 0;
        const char * mc = std::getenv("GGML_PCUDA_MATVEC_COLS");
        r.matvec_cols = mc && std::strcmp(mc, "1") == 0;
        if (const char * rb = std::getenv("GGML_PCUDA_ROW_BLOCK")) r.row_block = std::max(1, std::atoi(rb));
        if (const char * mc = std::getenv("GGML_PCUDA_MATMUL_CHUNKS")) r.matmul_chunks = std::clamp(std::atoi(mc), 1, 16);
        if (const char * qk = std::getenv("GGML_PCUDA_Q8K")) r.q8k = std::strcmp(qk, "0") != 0;
        if (const char * tl = std::getenv("GGML_PCUDA_TILE")) r.tile = std::atoi(tl) == 4 ? 4 : 8;
        r.ok = true;
        GGML_LOG_INFO("pcuda: %s, pool of %u threads + caller (%u for bandwidth-bound launches), graphs %s\n", p.name,
                      p.worker_count, r.fast_workers, r.graphs ? "on" : "off");
        return r;
    }();
    return rt;
}

// ---- weight streams ---------------------------------------------------------------

bool is_kquant(ggml_type t) { return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K; }
int  stream_words(ggml_type t) { return t == GGML_TYPE_Q4_K ? 37 : 53; }  // per row and 256 weights

bool weight_shape_ok(const ggml_tensor * w) {
    return is_kquant(w->type) && w->ne[2] == 1 && w->ne[3] == 1 && w->ne[0] % 256 == 0 && w->ne[1] % 8 == 0 &&
           w->ne[0] <= INT32_MAX && w->ne[1] <= INT32_MAX && ggml_is_contiguous(w);
}

size_t stream_bytes(const ggml_tensor * w) {
    return size_t(w->ne[1]) * size_t(w->ne[0] / 256) * size_t(stream_words(w->type)) * 4;
}

uint16_t f32_to_f16_exact(float f) {  // the value came from an fp16, so this is exact
    return ggml_fp32_to_fp16(f);
}

// Inverse of ref::repackQ4K / repackQ6K: interleaved stream -> ggml blocks.
void unpack_stream(const ggml_tensor * w, uint8_t * out) {
    const int rows = int(w->ne[1]), k = int(w->ne[0]), nsb = k / 256, words = stream_words(w->type);
    const int bb = w->type == GGML_TYPE_Q4_K ? 144 : 210;
    const int * qw = static_cast<const int *>(w->data);
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        for (int s = 0; s < nsb; ++s) {
            const int * blk = qw + (size_t(g) * nsb + s) * size_t(words) * 8;
            uint8_t * b = out + (size_t(row) * nsb + s) * bb;
            auto word = [&](int i) { return blk[i * 8 + l]; };
            auto f32 = [&](int i) {
                const int v = word(i);
                float f;
                std::memcpy(&f, &v, 4);
                return f;
            };
            if (w->type == GGML_TYPE_Q4_K) {
                const uint16_t d = f32_to_f16_exact(f32(0)), dmin = f32_to_f16_exact(f32(1));
                std::memcpy(b, &d, 2);
                std::memcpy(b + 2, &dmin, 2);
                for (int i = 0; i < 3; ++i) { const int v = word(2 + i); std::memcpy(b + 4 + 4 * i, &v, 4); }
                for (int i = 0; i < 32; ++i) { const int v = word(5 + i); std::memcpy(b + 16 + 4 * i, &v, 4); }
            } else {
                for (int i = 0; i < 52; ++i) { const int v = word(i); std::memcpy(b + 4 * i, &v, 4); }
                const uint16_t d = f32_to_f16_exact(f32(52));
                std::memcpy(b + 208, &d, 2);
            }
        }
    }
}

// ---- buffer ---------------------------------------------------------------------------

struct BufferCtx {
    void * base = nullptr;
    size_t size = 0;
};

void buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<BufferCtx *>(buffer->context);
    pcudaFree(ctx->base);
    delete ctx;
}

void * buffer_get_base(ggml_backend_buffer_t buffer) { return static_cast<BufferCtx *>(buffer->context)->base; }

void buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

void buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (is_kquant(tensor->type) && weight_shape_ok(tensor)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor) && "pcuda: K-quant weights are uploaded whole");
        const auto * src = static_cast<const uint8_t *>(data);
        int * dst = static_cast<int *>(tensor->data);
        if (tensor->type == GGML_TYPE_Q4_K) llm::ref::repackQ4K(src, int(tensor->ne[1]), int(tensor->ne[0]), dst);
        else llm::ref::repackQ6K(src, int(tensor->ne[1]), int(tensor->ne[0]), dst);
        return;
    }
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (is_kquant(tensor->type) && weight_shape_ok(tensor)) {
        std::vector<uint8_t> full(ggml_nbytes(tensor));
        unpack_stream(tensor, full.data());
        std::memcpy(data, full.data() + offset, size);
        return;
    }
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<BufferCtx *>(buffer->context);
    std::memset(ctx->base, value, ctx->size);
}

const ggml_backend_buffer_i buffer_i = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buffer_memset_tensor,
    /* .set_tensor    = */ buffer_set_tensor,
    /* .get_tensor    = */ buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ buffer_clear,
    /* .reset         = */ nullptr,
};

// ---- buffer type ------------------------------------------------------------------------

const char * buft_get_name(ggml_backend_buffer_type_t) { return "PCUDA"; }

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!runtime().ok) return nullptr;
    void * p = nullptr;
    const size_t bytes = size ? size : 64;  // llama.cpp probes with empty buffers
    if (pcudaMalloc(&p, bytes) != pcuda::Error::Success) {
        GGML_LOG_ERROR("pcuda: allocating %zu bytes failed: %s\n", bytes, pcudaGetLastErrorDetail());
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, buffer_i, new BufferCtx{p, bytes}, size);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }

size_t buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    if (is_kquant(tensor->type) && weight_shape_ok(tensor)) return std::max(stream_bytes(tensor), ggml_nbytes(tensor));
    return ggml_nbytes(tensor);
}

bool buft_is_host(ggml_backend_buffer_type_t) { return false; }  // repacked weights are not ggml layout

const ggml_backend_buffer_type_i buft_i = {
    /* .get_name         = */ buft_get_name,
    /* .alloc_buffer     = */ buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ buft_is_host,
};

ggml_backend_buffer_type_t pcuda_buffer_type();

// ---- backend ------------------------------------------------------------------------------

struct BackendCtx {
    int *                             scratch = nullptr;  // 8-bit activations, one slot per column
    size_t                            scratch_ints = 0;
    size_t                            scratch_used = 0;
    std::map<std::vector<int>, int *> consts;             // device int arrays (dims) by value
    std::map<int, float *>            ones;               // device float[n] of 1.0f
    bool                              capturing = false;
    struct Quantized {  // inputs quantized in the current recording (by tensor: memory is reused)
        const ggml_tensor * t;
        int64_t      ncols;
        int          k;
        bool         k256;  // quantize_cols_k slots (one scale per 256 values)
        int *        q;
    };
    std::vector<Quantized>            quantized;
    // GGML_PCUDA_PROFILE=1: time spent in graph_compute, recording and kernel runs
    bool                              profile = false;
    double                            t_compute = 0, t_record = 0, t_run = 0;
    long long                         calls = 0, launches = 0;
    std::map<std::string, std::pair<double, long long>> per_kernel;  // graphs off: synchronous launches, timed

    ~BackendCtx() {
        if (profile && calls)
            std::fprintf(stderr, "pcuda profile: %lld graph_compute calls, %lld launches; compute %.1f ms, recording %.1f ms, "
                          "kernel runs %.1f ms\n", calls, launches, t_compute * 1e3, t_record * 1e3, t_run * 1e3);
        if (profile)
            for (const auto & [name, tc] : per_kernel)
                std::fprintf(stderr, "  %-16s %8lld launches %9.1f ms %7.1f us each\n", name.c_str(), tc.second, tc.first * 1e3,
                             tc.first * 1e6 / double(tc.second));
        if (scratch) pcudaFree(scratch);
        for (auto & [k, p] : consts) pcudaFree(p);
        for (auto & [k, p] : ones) pcudaFree(p);
    }
    const int * ints(const std::vector<int> & v) {
        auto & p = consts[v];
        if (!p) {
            void * m = nullptr;
            if (pcudaMalloc(&m, v.size() * sizeof(int)) != pcuda::Error::Success) return nullptr;
            std::memcpy(m, v.data(), v.size() * sizeof(int));  // pcuda memory is host memory (and copies are refused while recording)
            p = static_cast<int *>(m);
        }
        return p;
    }
    const float * ones_of(int n) {
        auto & p = ones[n];
        if (!p) {
            void * m = nullptr;
            if (pcudaMalloc(&m, size_t(n) * sizeof(float)) != pcuda::Error::Success) return nullptr;
            const std::vector<float> v(size_t(n), 1.0f);
            std::memcpy(m, v.data(), v.size() * sizeof(float));
            p = static_cast<float *>(m);
        }
        return p;
    }
};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool flush(BackendCtx & ctx) {  // run the recorded launches
    if (!ctx.capturing) return true;
    ctx.capturing = false;
    pcudaGraph_t g = nullptr;
    if (pcudaStreamEndCapture(nullptr, &g) != pcuda::Error::Success) return false;
    const double t0 = ctx.profile ? now() : 0;
    const pcuda::Error e = pcudaGraphLaunch(g, nullptr);
    if (ctx.profile) ctx.t_run += now() - t0;
    if (e != pcuda::Error::Success) GGML_LOG_ERROR("pcuda: graph launch failed: %s\n", pcudaGetLastErrorDetail());
    pcudaGraphDestroy(g);
    return e == pcuda::Error::Success;
}

bool begin(BackendCtx & ctx) {  // start recording (graphs on) with an empty scratch
    if (ctx.capturing) return true;
    ctx.scratch_used = 0;
    ctx.quantized.clear();
    if (!runtime().graphs) return true;
    if (pcudaStreamBeginCapture(nullptr) != pcuda::Error::Success) return false;
    ctx.capturing = true;
    return true;
}

template <class Args>
bool launch(BackendCtx & ctx, const char * kernel, const Args & a, int64_t min_chunk, unsigned workers = 0) {
    const double t0 = ctx.profile && !ctx.capturing ? now() : 0;
    const pcuda::Error e = min_chunk ? pcudaLaunchKernelChunked(kernel, &a, workers ? workers : runtime().fast_workers, min_chunk)
                                     : pcudaLaunchKernel(kernel, &a);
    if (e != pcuda::Error::Success) {
        GGML_LOG_ERROR("pcuda: %s failed: %s\n", kernel, pcudaGetLastErrorDetail());
        return false;
    }
    if (ctx.profile) ++ctx.launches;
    if (ctx.profile && !ctx.capturing) {
        auto & tc = ctx.per_kernel[kernel];
        tc.first += now() - t0;
        ++tc.second;
    }
    return true;
}

// The input's columns as 8-bit activations in the scratch (quantize_q8x4 runs in the recorded
// graph, reading the input in pcuda memory). Matrices that share an input (q, k, v; gate, up)
// quantize it once per recording.
float op_param_f32(const ggml_tensor * t, int i);
bool  is_row_vector_of(const ggml_tensor * v, const ggml_tensor * x);

// Batches whose matmuls read activations scaled per 256 values (quantize_cols_k slots).
bool batch_k256(int64_t ncols) { return ncols > 1 && !runtime().matvec_cols && runtime().q8k; }

// Scratch slots for the 8-bit form of tensor x (ncols columns of k values), registered so that the
// matmuls reading x find it, then extra ints for the caller. The caller records the kernel that fills them.
int * quantized_slots(BackendCtx & ctx, const ggml_tensor * x, int k, int64_t ncols, bool k256 = false, size_t extra = 0) {
    const size_t slot = size_t(k256 ? llm::scratchKInts(k / 256) : llm::scratchInts(k / 32));
    const size_t total = slot * size_t(ncols) + extra;
    if (ctx.scratch_used + total > ctx.scratch_ints) {  // grow: run what is recorded first
        if (!flush(ctx)) return nullptr;
        const size_t need = std::max(total * 4, ctx.scratch_ints * 2);
        if (ctx.scratch) pcudaFree(ctx.scratch);
        ctx.scratch = nullptr;
        void * p = nullptr;
        if (pcudaMalloc(&p, need * sizeof(int)) != pcuda::Error::Success) return nullptr;
        ctx.scratch = static_cast<int *>(p);
        ctx.scratch_ints = need;
        if (!begin(ctx)) return nullptr;
    }
    int * base = ctx.scratch + ctx.scratch_used;
    ctx.scratch_used += total;
    ctx.quantized.push_back({x, ncols, k, k256, base});
    return base;
}

// k256: one scale per 256 values (quantize_cols_k; contiguous columns only).
int * quantized_input(BackendCtx & ctx, const ggml_tensor * x, int k, int64_t ncols, bool k256 = false) {
    const int nb = k / 32;
    const size_t slot = size_t(llm::scratchInts(nb));
    for (const auto & qi : ctx.quantized)
        if (qi.t == x && qi.ncols == ncols && qi.k == k && qi.k256 == k256) return qi.q;
    int * base = quantized_slots(ctx, x, k, ncols, k256);
    if (!base) return nullptr;
    if (k256) {
        const int * dims = ctx.ints({k / 256, int(ncols)});
        if (!dims) return nullptr;
        const llm::QuantizeColsArgs a{static_cast<const float *>(x->data), base, dims, k, int(ncols)};
        return launch(ctx, "quantize_cols_k", a, 8) ? base : nullptr;
    }
    if (ncols > 1 && x->nb[1] == size_t(k) * sizeof(float)) {  // contiguous columns: one launch
        const int * dims = ctx.ints({nb, int(ncols)});
        if (!dims) return nullptr;
        const llm::QuantizeColsArgs a{static_cast<const float *>(x->data), base, dims, k, int(ncols)};
        return launch(ctx, "quantize_cols", a, 16) ? base : nullptr;
    }
    for (int64_t c = 0; c < ncols; ++c) {
        int * q = base + size_t(c) * slot;
        const auto * xc = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + c * x->nb[1]);
        // up to 128 blocks (4096 values) is less work than waking the pool
        if (!launch(ctx, "quantize_q8x4", llm::QuantizeArgs{xc, q, llm::scratchScales(q, nb), nb}, nb > 128 ? 16 : 0)) return nullptr;
    }
    return base;
}

// ADD (optional) + RMS_NORM + MUL by the norm weight, in two parallel kernels, with the result also
// quantized for the matmuls that read it (batches: per 256 values, the partial sums of squares
// after the slots). n % 256 == 0.
bool fused_norm(BackendCtx & ctx, ggml_tensor * add, ggml_tensor * norm, ggml_tensor * mul) {
    const ggml_tensor * x = norm->src[0];
    const int n = int(x->ne[0]), rows = int(ggml_nrows(x));
    const bool k256 = batch_k256(rows);
    const int  off = k256 ? rows * llm::scratchKInts(n / 256) : 0;
    const int * dims = ctx.ints({n / 32, n, rows});
    const int * dimsk = k256 ? ctx.ints({n / 256, n, rows, off}) : dims;
    int * qk = dims && dimsk ? quantized_slots(ctx, mul, n, rows, k256, k256 ? size_t(rows) * llm::scratchInts(n / 32) : 0) : nullptr;
    if (!qk) return false;
    int * q = qk + off;  // the partial sums (and, per 32, the quantized slots)
    if (add) {
        const llm::AddSumsqFArgs a{static_cast<const float *>(add->src[0]->data), static_cast<const float *>(add->src[1]->data),
                                   static_cast<float *>(add->data), q, dims, n, rows};
        if (!launch(ctx, "add_sumsq_f", a, 16)) return false;
    } else {
        if (!launch(ctx, "sumsq_part", llm::SumsqPartArgs{static_cast<const float *>(x->data), q, dims, n, rows}, 16)) return false;
    }
    const llm::NormMulQ8Args nm{static_cast<const float *>(x->data), static_cast<const float *>(mul->src[1]->data),
                                static_cast<float *>(mul->data), qk, dimsk, n, rows, op_param_f32(norm, 0)};
    return k256 ? launch(ctx, "norm_mul_q8k_f", nm, 8) : launch(ctx, "norm_mul_q8_f", nm, 16);
}

// SwiGLU with the result also quantized for the down projection (batches: per 256 values). ne0 % 256 == 0.
bool fused_swiglu(BackendCtx & ctx, ggml_tensor * dst) {
    const int n = int(dst->ne[0]), cols = int(ggml_nrows(dst));
    const bool k256 = batch_k256(cols);
    const int * dims = ctx.ints({n, cols});
    int * q = dims ? quantized_slots(ctx, dst, n, cols, k256) : nullptr;
    if (!q) return false;
    const llm::SiluMulQ8FArgs a{static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
                                static_cast<float *>(dst->data), q, dims, n, cols};
    return k256 ? launch(ctx, "silu_mul_q8kf", a, 8) : launch(ctx, "silu_mul_q8x4f", a, 16);
}

bool mul_mat(BackendCtx & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int rows = int(w->ne[1]), k = int(w->ne[0]), nb = k / 32;
    const int64_t ncols = ggml_nrows(x);
    const int * dims = ctx.ints({rows, k, int(ncols), runtime().row_block});
    // Batches: activations scaled per 256 values (integer sums over whole super-blocks), unless off.
    const bool k256 = batch_k256(ncols) && x->nb[1] == size_t(k) * sizeof(float);
    int * base = dims ? quantized_input(ctx, x, k, ncols, k256) : nullptr;
    if (!base) return false;
    if (ncols > 1 && !runtime().matvec_cols) {  // several tokens: each weight word read once per 4 or 8 columns
        const bool  q4 = w->type == GGML_TYPE_Q4_K, x8 = runtime().tile == 8 && ncols > 4;
        const char * kernel = k256 ? (x8 ? (q4 ? "matmul_q4k_q8kx8" : "matmul_q6k_q8kx8") : (q4 ? "matmul_q4k_q8kx4" : "matmul_q6k_q8kx4"))
                                   : (x8 ? (q4 ? "matmul_q4k_q8x8" : "matmul_q6k_q8x8") : (q4 ? "matmul_q4k_q8x4" : "matmul_q6k_q8x4"));
        const llm::MatmulKQ8Args a{static_cast<const int *>(w->data), base, static_cast<float *>(dst->data), dims,
                                   rows, k, int(ncols), runtime().row_block};
        return launch(ctx, kernel, a, 16, runtime().workers * unsigned(runtime().matmul_chunks));
    }
    const char * kernel = w->type == GGML_TYPE_Q4_K ? "matvec_q4k_q8" : "matvec_q6k_q8";
    for (int64_t c = 0; c < ncols; ++c) {
        int * q = base + size_t(c) * size_t(llm::scratchInts(nb));
        const llm::MatvecQ4Q8Args a{static_cast<const int *>(w->data), q, llm::scratchScales(q, nb),
                                    static_cast<float *>(dst->data) + c * rows, dims, rows, k};
        if (!launch(ctx, kernel, a, 16)) return false;
    }
    return true;
}

float op_param_f32(const ggml_tensor * t, int i) {
    float v;
    std::memcpy(&v, reinterpret_cast<const int32_t *>(t->op_params) + i, sizeof v);
    return v;
}

// RMS_NORM (optionally fused with the following MUL by a row vector: the norm weight).
bool rms_norm(BackendCtx & ctx, const ggml_tensor * norm, const float * weight, ggml_tensor * out) {
    const ggml_tensor * x = norm->src[0];
    const int n = int(x->ne[0]), rows = int(ggml_nrows(x));
    const float * w = weight ? weight : ctx.ones_of(n);
    if (!w) return false;
    const llm::RmsNormArgs a{static_cast<const float *>(x->data), w, static_cast<float *>(out->data), n, rows,
                             op_param_f32(norm, 0)};
    return launch(ctx, "rms_norm", a, rows > 1 ? 1 : 0);
}

bool is_row_vector_of(const ggml_tensor * v, const ggml_tensor * x) {
    return v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) && v->ne[0] == x->ne[0] && ggml_nrows(v) == 1;
}

bool mul(BackendCtx & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const int n = is_row_vector_of(b, a) ? int(a->ne[0]) : int(ggml_nelements(a));  // broadcast or same shape
    const llm::MulRowsArgs m{static_cast<const float *>(a->data), static_cast<const float *>(b->data),
                             static_cast<float *>(dst->data), n, int(ggml_nelements(a))};
    return launch(ctx, "mul_rows", m, 0);
}

bool add(BackendCtx & ctx, ggml_tensor * dst) {
    const pcuda::kernels::VectorAddArgs a{static_cast<const float *>(dst->src[0]->data),
                                          static_cast<const float *>(dst->src[1]->data),
                                          static_cast<float *>(dst->data), int(ggml_nelements(dst))};
    return launch(ctx, "vector_add", a, a.count > 4096 ? 1024 : 0);
}

bool rope(BackendCtx & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const int hd = int(x->ne[0]), nh = int(x->ne[1]), nt = int(x->ne[2]), pairs = hd / 2;
    const float * ff = dst->src[2] ? static_cast<const float *>(dst->src[2]->data) : ctx.ones_of(pairs);
    const int * dims = ctx.ints({pairs, nh, pairs * nt});
    if (!ff || !dims) return false;
    const llm::RopeGgmlArgs a{static_cast<const float *>(x->data), static_cast<float *>(dst->data),
                              static_cast<const int *>(dst->src[1]->data), ff, dims, hd, nh, nt,
                              std::log2(op_param_f32(dst, 5))};
    return launch(ctx, "rope_ggml", a, 0);
}

bool swiglu(BackendCtx & ctx, ggml_tensor * dst) {
    const llm::SiluMulArgs a{static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
                             static_cast<float *>(dst->data), int(ggml_nelements(dst))};
    return launch(ctx, "silu_mul", a, 1024);
}

const char * backend_get_name(ggml_backend_t) { return "PCUDA"; }

void backend_free(ggml_backend_t backend) {
    delete static_cast<BackendCtx *>(backend->context);
    delete backend;
}

ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto & ctx = *static_cast<BackendCtx *>(backend->context);
    const double t0 = ctx.profile ? now() : 0;
    struct Done {
        BackendCtx & c;
        double t0;
        ~Done() { if (c.profile) c.t_compute += now() - t0, ++c.calls; }
    } done{ctx, t0};
    if (!begin(ctx)) return GGML_STATUS_FAILED;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        bool ok = true;
        switch (node->op) {
            case GGML_OP_MUL_MAT: ok = mul_mat(ctx, node); break;
            case GGML_OP_ADD:
            case GGML_OP_RMS_NORM: {
                // [ADD +] RMS_NORM + MUL by a row vector (the norm weight): two parallel kernels that also
                // quantize the result for the matmuls (n % 256 == 0), else one-thread rms_norm.
                static const ggml_op fused[] = {GGML_OP_RMS_NORM, GGML_OP_MUL};
                const int ni = node->op == GGML_OP_ADD ? i + 1 : i;  // the RMS_NORM node
                ggml_tensor * norm = ni < cgraph->n_nodes ? cgraph->nodes[ni] : nullptr;
                ggml_tensor * mul = ni + 1 < cgraph->n_nodes ? cgraph->nodes[ni + 1] : nullptr;
                const bool pattern = norm && mul && norm->op == GGML_OP_RMS_NORM && ggml_can_fuse(cgraph, ni, fused, 2) &&
                                     mul->src[0] == norm && is_row_vector_of(mul->src[1], norm) &&
                                     norm->src[0]->ne[0] % 256 == 0 && ggml_is_contiguous(norm->src[0]) &&
                                     (node->op == GGML_OP_RMS_NORM || norm->src[0] == node);
                if (pattern) {
                    ok = fused_norm(ctx, node->op == GGML_OP_ADD ? node : nullptr, norm, mul);
                    i = ni + 1;
                } else if (node->op == GGML_OP_ADD) {
                    ok = add(ctx, node);
                } else {
                    static const ggml_op pair[] = {GGML_OP_RMS_NORM, GGML_OP_MUL};
                    ggml_tensor * next = i + 1 < cgraph->n_nodes ? cgraph->nodes[i + 1] : nullptr;
                    if (next && ggml_can_fuse(cgraph, i, pair, 2) && next->src[0] == node && is_row_vector_of(next->src[1], node)) {
                        ok = rms_norm(ctx, node, static_cast<const float *>(next->src[1]->data), next);
                        ++i;
                    } else {
                        ok = rms_norm(ctx, node, nullptr, node);
                    }
                }
                break;
            }
            case GGML_OP_MUL: ok = mul(ctx, node); break;
            case GGML_OP_ROPE: ok = rope(ctx, node); break;
            case GGML_OP_GLU: ok = node->ne[0] % 256 == 0 ? fused_swiglu(ctx, node) : swiglu(ctx, node); break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
        if (!ok) {
            flush(ctx);
            return GGML_STATUS_FAILED;
        }
    }
    if (ctx.profile) ctx.t_record += now() - t0;
    return flush(ctx) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

const ggml_backend_i backend_i = {
    /* .get_name            = */ backend_get_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x70, 0x63, 0x75, 0x64, 0x61, 0x2d, 0x6b, 0x71, 0x75, 0x61, 0x6e, 0x74, 0x2d, 0x63, 0x70, 0x75};
    return &guid;
}

// ---- device ---------------------------------------------------------------------------------

const char * dev_get_name(ggml_backend_dev_t) { return "PCUDA"; }
const char * dev_get_description(ggml_backend_dev_t) { return "pcuda: CUDA C kernels on CPU cores (Q4_K/Q6_K matmul)"; }

void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = dev_get_name(dev);
    props->description = dev_get_description(dev);
    props->type = dev_get_type(dev);
    dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t, const char *) { return ggml_backend_pcuda_init(); }

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t) { return pcuda_buffer_type(); }

bool f32c(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

bool dev_supports_op_impl(ggml_backend_dev_t, const ggml_tensor * op);

bool dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool r = dev_supports_op_impl(dev, op);
    static const bool dbg = std::getenv("GGML_PCUDA_DEBUG") != nullptr;
    if (dbg && op->op != GGML_OP_NONE && op->op != GGML_OP_VIEW && op->op != GGML_OP_RESHAPE && op->op != GGML_OP_PERMUTE)
        std::fprintf(stderr, "pcuda supports_op %s %s: %d (src0 %s %s)\n", ggml_op_desc(op), op->name, r ? 1 : 0,
                     op->src[0] ? ggml_type_name(op->src[0]->type) : "-", op->src[0] && ggml_is_contiguous(op->src[0]) ? "contig" : "strided");
    return r;
}

bool dev_supports_op_impl(ggml_backend_dev_t, const ggml_tensor * op) {
    const PcudaRuntime & rt = runtime();
    if (!rt.ok || rt.disabled) return false;
    const ggml_tensor * s0 = op->src[0];
    const ggml_tensor * s1 = op->src[1];
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            // The weight must live in this backend's buffers (repacked), any batch size.
            return s0->buffer && ggml_backend_buffer_get_type(s0->buffer) == pcuda_buffer_type() && weight_shape_ok(s0) &&
                   f32c(s1) && s1->ne[0] == s0->ne[0] && f32c(op) && op->ne[0] == s0->ne[1] &&
                   ggml_nrows(op) == ggml_nrows(s1);
        default:
            break;
    }
    if (rt.matmul_only) return false;
    switch (op->op) {
        case GGML_OP_RMS_NORM:
            return f32c(s0) && f32c(op) && ggml_are_same_shape(s0, op);
        case GGML_OP_ADD:
            return f32c(s0) && f32c(s1) && f32c(op) && ggml_are_same_shape(s0, s1) && ggml_are_same_shape(s0, op);
        case GGML_OP_MUL:  // by a row vector (norm weight, n % 8 == 0) or elementwise
            return f32c(s0) && f32c(s1) && f32c(op) && ggml_are_same_shape(s0, op) &&
                   ((is_row_vector_of(s1, s0) && s0->ne[0] % 8 == 0) ||
                    (ggml_are_same_shape(s0, s1) && ggml_nelements(s0) % 8 == 0));
        case GGML_OP_ROPE: {
            const int32_t * p = reinterpret_cast<const int32_t *>(op->op_params);
            const ggml_tensor * ff = op->src[2];
            return p[2] == 0 /* NORM mode */ && p[1] == s0->ne[0] && op_param_f32(op, 6) == 1.0f && op_param_f32(op, 7) == 0.0f &&
                   op_param_f32(op, 8) == 1.0f && p[15] == 0 && f32c(s0) && f32c(op) && ggml_are_same_shape(s0, op) &&
                   s0->ne[0] % 16 == 0 && s0->ne[3] == 1 && s1 && s1->type == GGML_TYPE_I32 && ggml_is_contiguous(s1) &&
                   s1->ne[0] == s0->ne[2] && (!ff || (f32c(ff) && ff->ne[0] >= s0->ne[0] / 2));
        }
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(op, 1) == 0 && f32c(s0) && f32c(s1) &&
                   f32c(op) && ggml_are_same_shape(s0, s1) && ggml_are_same_shape(s0, op);
        default:
            return false;
    }
}

// Host buffers are not readable by the kernels (pcuda validates every pointer against its own
// allocations), so the scheduler copies such inputs (positions, the first embedding, the
// attention output) into this backend's buffers.
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == pcuda_buffer_type(); }

const ggml_backend_device_i device_i = {
    /* .get_name             = */ dev_get_name,
    /* .get_description      = */ dev_get_description,
    /* .get_memory           = */ dev_get_memory,
    /* .get_type             = */ dev_get_type,
    /* .get_props            = */ dev_get_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---- registry ------------------------------------------------------------------------------

const char * reg_get_name(ggml_backend_reg_t) { return "PCUDA"; }
size_t       reg_get_device_count(ggml_backend_reg_t) { return 1; }

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

void * reg_get_proc_address(ggml_backend_reg_t, const char *) { return nullptr; }

const ggml_backend_reg_i reg_i = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

ggml_backend_buffer_type_t pcuda_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ buft_i,
        /* .device  = */ reg_get_device(ggml_backend_pcuda_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

} // namespace

ggml_backend_t ggml_backend_pcuda_init(void) {
    if (!runtime().ok) return nullptr;
    return new ggml_backend{
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_i,
        /* .device  = */ reg_get_device(ggml_backend_pcuda_reg(), 0),
        /* .context = */ [] {
            auto * c = new BackendCtx;
            const char * p = std::getenv("GGML_PCUDA_PROFILE");
            c->profile = p && std::strcmp(p, "1") == 0;
            return c;
        }(),
    };
}

bool ggml_backend_is_pcuda(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, backend_guid());
}

ggml_backend_reg_t ggml_backend_pcuda_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_pcuda_reg)
