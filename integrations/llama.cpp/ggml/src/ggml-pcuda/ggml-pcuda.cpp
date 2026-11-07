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

#include "llm_kernels.hpp"
#include "llm_reference.hpp"

#include <chrono>
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
    unsigned workers = 1;
    bool     graphs = true;
    bool     disabled = false;  // GGML_PCUDA_DISABLE=1: claim nothing (CPU-only comparisons)
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
        const char * g = std::getenv("GGML_PCUDA_GRAPHS");
        r.graphs = !(g && std::strcmp(g, "0") == 0);
        const char * off = std::getenv("GGML_PCUDA_DISABLE");
        r.disabled = off && std::strcmp(off, "1") == 0;
        r.ok = true;
        GGML_LOG_INFO("pcuda: %s, pool of %u threads + caller, graphs %s\n", p.name, p.worker_count, r.graphs ? "on" : "off");
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
    std::map<std::pair<int, int>, int *> dims;            // device {rows, k} per weight shape
    bool                              capturing = false;
    struct Quantized {                                    // inputs quantized in the current recording
        const void * data;
        int64_t      ncols;
        int          k;
        int *        q;
    };
    std::vector<Quantized>            quantized;
    // GGML_PCUDA_PROFILE=1: time spent in graph_compute, host quantization and kernel runs
    bool                              profile = false;
    double                            t_compute = 0, t_quant = 0, t_run = 0;
    long long                         calls = 0, launches = 0;

    ~BackendCtx() {
        if (profile && calls)
            std::fprintf(stderr, "pcuda profile: %lld graph_compute calls, %lld launches; compute %.1f ms, input staging %.1f ms, "
                          "kernel runs %.1f ms\n", calls, launches, t_compute * 1e3, t_quant * 1e3, t_run * 1e3);
        if (scratch) pcudaFree(scratch);
        for (auto & [k, p] : dims) pcudaFree(p);
    }
    int * dims_for(int rows, int k) {
        auto & p = dims[{rows, k}];
        if (!p) {
            void * m = nullptr;
            if (pcudaMalloc(&m, 2 * sizeof(int)) != pcuda::Error::Success) return nullptr;
            const int v[2] = {rows, k};
            pcudaMemcpy(m, v, sizeof v, pcuda::CopyKind::HostToDevice);
            p = static_cast<int *>(m);
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

// The input's columns as 8-bit activations in the scratch. Each column slot holds the quantized
// words and scales (llm::scratchInts) followed by a float copy of the column: the copy is made now
// (the input is host memory outside pcuda), the quantize_q8x4 kernel runs in the recorded graph.
// Matrices that share an input (q, k, v; gate, up) quantize it once.
int * quantized_input(BackendCtx & ctx, const ggml_tensor * x, int k, int64_t ncols) {
    const int nb = k / 32;
    const size_t slot = size_t(llm::scratchInts(nb)) + size_t(k);
    for (const auto & qi : ctx.quantized)
        if (qi.data == x->data && qi.ncols == ncols && qi.k == k) return qi.q;
    if (ctx.scratch_used + slot * size_t(ncols) > ctx.scratch_ints) {  // grow: run what is recorded first
        if (!flush(ctx)) return nullptr;
        const size_t need = std::max(slot * size_t(ncols) * 4, ctx.scratch_ints * 2);
        if (ctx.scratch) pcudaFree(ctx.scratch);
        ctx.scratch = nullptr;
        void * p = nullptr;
        if (pcudaMalloc(&p, need * sizeof(int)) != pcuda::Error::Success) return nullptr;
        ctx.scratch = static_cast<int *>(p);
        ctx.scratch_ints = need;
        if (!begin(ctx)) return nullptr;
    }
    int * base = ctx.scratch + ctx.scratch_used;
    ctx.scratch_used += slot * size_t(ncols);
    const double tq = ctx.profile ? now() : 0;
    for (int64_t c = 0; c < ncols; ++c) {
        int * q = base + size_t(c) * slot;
        float * stage = reinterpret_cast<float *>(q + llm::scratchInts(nb));
        std::memcpy(stage, static_cast<const char *>(x->data) + c * x->nb[1], size_t(k) * sizeof(float));
        const llm::QuantizeArgs qa{stage, q, llm::scratchScales(q, nb), nb};
        if (pcudaLaunchKernelChunked("quantize_q8x4", &qa, runtime().workers, 16) != pcuda::Error::Success) {
            GGML_LOG_ERROR("pcuda: quantize_q8x4 failed: %s\n", pcudaGetLastErrorDetail());
            return nullptr;
        }
    }
    if (ctx.profile) ctx.t_quant += now() - tq;
    ctx.quantized.push_back({x->data, ncols, k, base});
    return base;
}

bool mul_mat(BackendCtx & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int rows = int(w->ne[1]), k = int(w->ne[0]), nb = k / 32;
    const int64_t ncols = ggml_nrows(x);
    const size_t slot = size_t(llm::scratchInts(nb)) + size_t(k);

    // Input produced by this backend in the same call: run what is recorded first.
    if (x->buffer && ggml_backend_buffer_get_type(x->buffer) == pcuda_buffer_type() && !flush(ctx)) return false;
    int * dims = ctx.dims_for(rows, k);
    if (!dims || !begin(ctx)) return false;
    int * base = quantized_input(ctx, x, k, ncols);
    if (!base) return false;
    const char * kernel = w->type == GGML_TYPE_Q4_K ? "matvec_q4k_q8" : "matvec_q6k_q8";
    for (int64_t c = 0; c < ncols; ++c) {
        int * q = base + size_t(c) * slot;
        const llm::MatvecQ4Q8Args a{static_cast<const int *>(w->data), q, llm::scratchScales(q, nb),
                                    static_cast<float *>(dst->data) + c * rows, dims, rows, k};
        if (pcudaLaunchKernelChunked(kernel, &a, runtime().workers, 16) != pcuda::Error::Success) {
            GGML_LOG_ERROR("pcuda: %s failed: %s\n", kernel, pcudaGetLastErrorDetail());
            return false;
        }
        if (ctx.profile) ++ctx.launches;
    }
    return true;
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
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!mul_mat(ctx, node)) {
                    flush(ctx);
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
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

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            if (runtime().disabled) return false;
            const ggml_tensor * w = op->src[0];
            const ggml_tensor * x = op->src[1];
            // The weight must live in this backend's buffers (repacked), any batch size.
            return w->buffer && ggml_backend_buffer_get_type(w->buffer) == pcuda_buffer_type() && weight_shape_ok(w) &&
                   x->type == GGML_TYPE_F32 && x->ne[0] == w->ne[0] && x->nb[0] == sizeof(float) &&
                   ggml_is_contiguous(x) && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                   op->ne[0] == w->ne[1] && ggml_nrows(op) == ggml_nrows(x) && runtime().ok;
        }
        default:
            return false;
    }
}

bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == pcuda_buffer_type() || ggml_backend_buft_is_host(buft);
}

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
