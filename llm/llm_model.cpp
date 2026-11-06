#include "llm_model.hpp"

#include "llm_reference.hpp"

#include <pcuda/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <stdexcept>
#include <string>

namespace llm {

// ---- configurations -----------------------------------------------------------

int64_t Config::weightParams() const {
    const int64_t qd = int64_t(heads) * head_dim;
    const int64_t per_layer = qd * hidden                     // wq
                              + 2 * int64_t(kvStride()) * hidden  // wk, wv
                              + int64_t(hidden) * qd           // wo
                              + 3 * int64_t(inter) * hidden;   // wg, wu, wd
    return per_layer * layers + int64_t(vocab) * hidden;
}

int64_t Config::weightBytes() const {
    const int64_t p = weightParams();
    const int64_t values = weights == Weights::Q8 ? p : p / 2;
    return values + p / 32 * int64_t(sizeof(float));
}

Config llama32_3b(int max_ctx, Weights w) {
    Config c;
    c.name = "Llama 3.2 3B shapes";
    c.hidden = 3072, c.inter = 8192, c.layers = 28, c.heads = 24, c.kv_heads = 8, c.head_dim = 128;
    c.vocab = 128256, c.max_ctx = max_ctx, c.weights = w;
    return c;
}

Config llama32_1b(int max_ctx, Weights w) {
    Config c;
    c.name = "Llama 3.2 1B shapes";
    c.hidden = 2048, c.inter = 8192, c.layers = 16, c.heads = 32, c.kv_heads = 8, c.head_dim = 64;
    c.vocab = 128256, c.max_ctx = max_ctx, c.weights = w;
    return c;
}

Config tinyConfig(Weights w) {
    Config c;
    c.name = "tiny";
    c.hidden = 256, c.inter = 640, c.layers = 2, c.heads = 4, c.kv_heads = 2, c.head_dim = 64;
    c.vocab = 1000, c.max_ctx = 40, c.weights = w;
    return c;
}

const char* weightsName(Weights w) {
    switch (w) {
    case Weights::Q8: return "Q8 (8-bit + f32 scale per 32)";
    case Weights::Q4: return "Q4 (4-bit + f32 scale per 32)";
    case Weights::Q4X8: return "Q4 x8 (4-bit + f32 scale per 32, 8 rows interleaved)";
    case Weights::Q4X8Q8: return "Q4 x8 with 8-bit activations (4-bit + f32 scale per 32; integer dot products)";
    }
    return "?";
}

const char* opName(Op op) {
    switch (op) {
    case Op::MatvecAttn: return "matvec attention (q,k,v,o)";
    case Op::MatvecFfn: return "matvec FFN (gate,up,down)";
    case Op::MatvecLmHead: return "matvec lm_head";
    case Op::RmsNorm: return "rms_norm (first layer)";
    case Op::Rope: return "rope + KV append";
    case Op::Attention: return "attention (scores,softmax,values)";
    case Op::SiluMul: return "silu_mul (+ quantize)";
    case Op::Add: return "residual add + rms_norm";
    case Op::Count: break;
    }
    return "?";
}

// ---- allocators ---------------------------------------------------------------

namespace {

[[noreturn]] void pcudaFail(const char* what, pcuda::Error e) {
    throw std::runtime_error(std::string(what) + ": " + pcudaGetErrorString(e) + " (" + pcudaGetLastErrorDetail() + ")");
}

void pcudaCheck(pcuda::Error e, const char* what) {
    if (e != pcuda::Error::Success) pcudaFail(what, e);
}

} // namespace

HostAllocator::~HostAllocator() {
    for (void* p : blocks_) _aligned_free(p);
}

void* HostAllocator::alloc(size_t bytes) {
    void* p = _aligned_malloc(bytes ? bytes : 1, 64);
    if (!p) throw std::runtime_error("host allocation failed");
    blocks_.push_back(p);
    return p;
}

void HostAllocator::upload(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }
void HostAllocator::download(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

PcudaAllocator::~PcudaAllocator() {
    for (void* p : blocks_) pcudaFree(p);
}

void* PcudaAllocator::alloc(size_t bytes) {
    void* p = nullptr;
    pcudaCheck(pcudaMalloc(&p, bytes ? bytes : 1), "pcudaMalloc");
    blocks_.push_back(p);
    return p;
}

void PcudaAllocator::upload(void* dst, const void* src, size_t bytes) {
    pcudaCheck(pcudaMemcpy(dst, src, bytes, pcuda::CopyKind::HostToDevice), "pcudaMemcpy H2D");
}

void PcudaAllocator::download(void* dst, const void* src, size_t bytes) {
    pcudaCheck(pcudaMemcpy(dst, src, bytes, pcuda::CopyKind::DeviceToHost), "pcudaMemcpy D2H");
}

// ---- deterministic model data -------------------------------------------------

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uniform() { return static_cast<float>(next() >> 40) * (2.0f / 16777216.0f) - 1.0f; }  // [-1, 1)
};

template <class T>
std::vector<T*> uploadAll(const std::vector<Allocator*>& allocs, const std::vector<T>& host) {
    std::vector<T*> out;
    for (Allocator* a : allocs) {
        void* p = a->alloc(host.size() * sizeof(T));
        a->upload(p, host.data(), host.size() * sizeof(T));
        out.push_back(static_cast<T*>(p));
    }
    return out;
}

std::vector<float> uniformVector(Rng& rng, size_t n, float center, float spread) {
    std::vector<float> v(n);
    for (float& f : v) f = center + spread * rng.uniform();
    return v;
}

void randomBytes(Rng& rng, std::vector<uint8_t>& v) {
    for (size_t i = 0; i < v.size(); i += 8) {
        const uint64_t r = rng.next();
        for (size_t j = 0; j < 8 && i + j < v.size(); ++j) v[i + j] = static_cast<uint8_t>(r >> (8 * j));
    }
}

// Scales keep y = Wx at about unit variance. Q8: int8 uniform in [-127, 127] (variance 127^2/3);
// Q4: nibble - 8 uniform in [-8, 7] (variance 21.25).
std::vector<Mat> makeMat(Rng& rng, const std::vector<Allocator*>& allocs, int rows, int k, Weights fmt) {
    const size_t n = size_t(rows) * size_t(k);
    std::vector<uint8_t> q(fmt == Weights::Q8 ? n : n / 2);
    randomBytes(rng, q);
    float var = 21.25f;
    if (fmt == Weights::Q8) {
        for (auto& b : q)
            if (b == 0x80) b = 0x81;  // -128 -> -127
        var = 127.0f * 127.0f / 3.0f;
    }
    const float base = 1.0f / std::sqrt(var * static_cast<float>(k));
    std::vector<float> s = uniformVector(rng, n / 32, base, 0.1f * base);
    if (fmt == Weights::Q4X8Q8) {  // same values as Q4, in the interleaved int stream with scales
        if (rows % 8 != 0) throw std::invalid_argument("Q4X8Q8 needs a multiple of 8 rows");
        std::vector<int> qw(n * 5 / 32);
        ref::repackQ4x8Q8(q.data(), s.data(), qw.data(), rows, k);
        const std::vector<int> dims{rows, k};
        const auto qd = uploadAll(allocs, qw);
        const auto dd = uploadAll(allocs, dims);
        std::vector<Mat> out;
        for (size_t i = 0; i < allocs.size(); ++i) out.push_back(Mat{rows, k, fmt, qd[i], nullptr, dd[i]});
        return out;
    }
    if (fmt == Weights::Q4X8) {  // same values as Q4, repacked
        if (rows % 8 != 0) throw std::invalid_argument("Q4X8 needs a multiple of 8 rows");
        std::vector<uint8_t> q8(q.size());
        std::vector<float> s8(s.size());
        ref::repackQ4x8(q.data(), s.data(), q8.data(), s8.data(), rows, k);
        q.swap(q8);
        s.swap(s8);
    }
    const std::vector<int> dims{rows, k};
    const auto qd = uploadAll(allocs, q);
    const auto sd = uploadAll(allocs, s);
    const auto dd = uploadAll(allocs, dims);
    std::vector<Mat> out;
    for (size_t i = 0; i < allocs.size(); ++i) out.push_back(Mat{rows, k, fmt, qd[i], sd[i], dd[i]});
    return out;
}

} // namespace

void buildModels(const Config& c, uint64_t seed, const std::vector<Allocator*>& allocs, std::vector<Model>& out) {
    if (c.hidden % 32 || c.inter % 32 || (c.heads * c.head_dim) % 32 || c.head_dim % 2 || c.heads % c.kv_heads ||
        c.head_dim % 8 || c.max_ctx < 1)
        throw std::invalid_argument("model shapes must be multiples of 32 with heads divisible by kv_heads");
    Rng rng{seed | 1};
    const size_t na = allocs.size();
    out.assign(na, Model{});
    for (Model& m : out) {
        m.cfg = c;
        m.layers.resize(size_t(c.layers));
    }
    const int qd = c.heads * c.head_dim;
    const Weights f = c.weights;
    for (int l = 0; l < c.layers; ++l) {
        auto wqkv = makeMat(rng, allocs, qd + 2 * c.kvStride(), c.hidden, f);
        auto wo = makeMat(rng, allocs, c.hidden, qd, f);
        auto wgu = makeMat(rng, allocs, 2 * c.inter, c.hidden, f);
        auto wd = makeMat(rng, allocs, c.hidden, c.inter, f);
        const auto an = uploadAll(allocs, uniformVector(rng, size_t(c.hidden), 1.0f, 0.1f));
        const auto fn = uploadAll(allocs, uniformVector(rng, size_t(c.hidden), 1.0f, 0.1f));
        const auto kc = uploadAll(allocs, uniformVector(rng, size_t(c.kStride()) * size_t(c.kvStride()), 0.0f, 1.0f));
        const auto vc = uploadAll(allocs, uniformVector(rng, size_t(c.max_ctx) * size_t(c.kvStride()), 0.0f, 1.0f));
        for (size_t i = 0; i < na; ++i) {
            Layer& L = out[i].layers[size_t(l)];
            L = Layer{wqkv[i], wo[i], wgu[i], wd[i], an[i], fn[i], kc[i], vc[i]};
        }
    }
    const auto on = uploadAll(allocs, uniformVector(rng, size_t(c.hidden), 1.0f, 0.1f));
    auto head = makeMat(rng, allocs, c.vocab, c.hidden, f);
    const AttnDims d = c.attnDims(c.max_ctx);
    const auto ad = uploadAll(allocs, std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad,
                                                       d.k_stride});
    for (size_t i = 0; i < na; ++i) {
        out[i].out_norm = on[i];
        out[i].lm_head = head[i];
        out[i].attn_dims = ad[i];
    }
}

Acts allocActs(const Config& c, Allocator& alloc, uint64_t seed) {
    auto buf = [&](int n) {
        const std::vector<float> zero(size_t(n), 0.0f);
        void* p = alloc.alloc(zero.size() * sizeof(float));
        alloc.upload(p, zero.data(), zero.size() * sizeof(float));
        return static_cast<float*>(p);
    };
    Acts a;
    a.x0 = buf(c.hidden);
    Rng rng{seed | 1};
    const std::vector<float> x0 = uniformVector(rng, size_t(c.hidden), 0.0f, 1.0f);
    alloc.upload(a.x0, x0.data(), x0.size() * sizeof(float));
    a.x = buf(c.hidden);
    a.xn = buf(c.hidden);
    a.q = buf(c.heads * c.head_dim + 2 * c.kvStride());
    a.k = a.q + c.heads * c.head_dim;
    a.v = a.k + c.kvStride();
    a.scores = buf(c.heads * padTo8(c.max_ctx));
    a.attn = buf(c.heads * c.head_dim);
    a.t = buf(c.hidden);
    a.g = buf(2 * c.inter);
    a.u = a.g + c.inter;
    a.h = buf(c.inter);
    a.logits = buf(c.vocab);
    const int maxk = std::max({c.hidden, c.inter, c.heads * c.head_dim});
    a.xq = reinterpret_cast<int*>(buf(scratchInts(maxk / 32)));
    return a;
}

// ---- decode step --------------------------------------------------------------

void decodeToken(Backend& b, const Model& m, const Acts& a, int pos, Profiler* prof) {
    const Config& c = m.cfg;
    if (pos < 0 || pos >= c.max_ctx) throw std::out_of_range("decode position outside the KV cache");
    auto op = [&](Op o, auto&& f) {
        if (prof) prof->begin(o);
        f();
        if (prof) prof->end(o);
    };
    b.setScratch(a.xq);
    b.setContext(pos + 1);
    b.copy(a.x, a.x0, c.hidden);
    b.beginToken(prof != nullptr);
    const bool q8 = c.weights == Weights::Q4X8Q8;
    op(Op::RmsNorm, [&] { b.rmsNorm(a.x, m.layers[0].attn_norm, a.xn); });
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const Layer& L = m.layers[l];
        op(Op::MatvecAttn, [&] {  // q, k and v in one launch
            if (q8 && l > 0) b.matvecQuantized(L.wqkv, a.q);  // input quantized by the last norm
            else b.matvec(L.wqkv, a.xn, a.q);
        });
        op(Op::Rope, [&] {
            b.rope(a.q, c.heads + c.kv_heads, pos);  // q and k are adjacent: one launch
            b.appendKv(a.k, a.v, L.k_cache, L.v_cache, pos);  // this token's K and V
        });
        op(Op::Attention, [&] {
            b.attnScores(a.q, L.k_cache, a.scores);
            b.softmax(a.scores);
            b.attnValues(a.scores, L.v_cache, a.attn);
        });
        op(Op::MatvecAttn, [&] { b.matvec(L.wo, a.attn, a.t); });
        if (q8) {
            op(Op::Add, [&] { b.addRmsNormQuant(a.x, a.t, L.ffn_norm); });
            op(Op::MatvecFfn, [&] { b.matvecQuantized(L.wgu, a.g); });  // gate and up in one launch
        } else {
            op(Op::Add, [&] { b.addRmsNorm(a.x, a.t, L.ffn_norm, a.xn); });
            op(Op::MatvecFfn, [&] { b.matvec(L.wgu, a.xn, a.g); });
        }
        if (q8) {
            op(Op::SiluMul, [&] { b.siluMulQuant(a.g, a.u, c.inter); });
            op(Op::MatvecFfn, [&] { b.matvecQuantized(L.wd, a.t); });
        } else {
            op(Op::SiluMul, [&] { b.siluMul(a.g, a.u, a.h, c.inter); });
            op(Op::MatvecFfn, [&] { b.matvec(L.wd, a.h, a.t); });
        }
        const float* next_norm = l + 1 < m.layers.size() ? m.layers[l + 1].attn_norm : m.out_norm;
        if (q8) op(Op::Add, [&] { b.addRmsNormQuant(a.x, a.t, next_norm); });
        else op(Op::Add, [&] { b.addRmsNorm(a.x, a.t, next_norm, a.xn); });
    }
    op(Op::MatvecLmHead, [&] {
        if (q8) b.matvecQuantized(m.lm_head, a.logits);
        else b.matvec(m.lm_head, a.xn, a.logits);
    });
    b.endToken();
}

// ---- backends -----------------------------------------------------------------

namespace {

class PcudaBackend final : public Backend {
public:
    PcudaBackend(const Model& m, unsigned workers, bool graphs, bool two_rows)
        : m_(m), workers_(workers), graphs_(graphs), two_rows_(two_rows), ctx_(m.cfg.max_ctx) {}
    ~PcudaBackend() override { abandonCapture(); }
    void abandonCapture() noexcept {
        if (!capturing_) return;
        capturing_ = false;
        pcudaGraph_t g = nullptr;
        if (pcudaStreamEndCapture(nullptr, &g) == pcuda::Error::Success) pcudaGraphDestroy(g);
    }

    void setScratch(int* q) override { xq_ = q; }
    void setContext(int ctx) override {
        ctx_ = ctx;
        const AttnDims d = m_.cfg.attnDims(ctx);
        const int dims[7] = {d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride};
        pcudaCheck(pcudaMemcpy(m_.attn_dims, dims, sizeof dims, pcuda::CopyKind::HostToDevice), "pcudaMemcpy dims");
    }
    void matvec(const Mat& w, const float* x, float* y) override {
        if (w.format == Weights::Q4X8Q8) {
            const QuantizeArgs qa{x, xq_, scratchScales(xq_, w.k / 32), w.k / 32};
            launch("quantize_q8x4", &qa, 16);
            matvecQuantized(w, y);
        } else if (w.format == Weights::Q4 || w.format == Weights::Q4X8) {
            const MatvecQ4Args a{static_cast<const uint8_t*>(w.q), w.s, x, y, w.dims, w.rows, w.k};
            launch(w.format == Weights::Q4 ? "matvec_q4" : "matvec_q4x8", &a, 16);
        } else {
            const MatvecQ8Args a{static_cast<const int8_t*>(w.q), w.s, x, y, w.dims, w.rows, w.k};
            launch("matvec_q8", &a, 16);
        }
    }
    void rmsNorm(const float* x, const float* w, float* y) override {
        const RmsNormArgs a{x, w, y, m_.cfg.hidden, 1, m_.cfg.eps};
        launch("rms_norm", &a, 0);
    }
    void addRmsNorm(float* x, const float* t, const float* w, float* y) override {
        const AddRmsNormArgs a{x, t, w, y, m_.cfg.hidden, m_.cfg.eps};
        launch("add_rms_norm", &a, 0);
    }
    void siluMulQuant(const float* g, const float* u, int n) override {
        const SiluMulQ8Args a{g, u, xq_, scratchScales(xq_, n / 32), n / 32};
        launch("silu_mul_q8x4", &a, 16);
    }
    void matvecQuantized(const Mat& w, float* y) override {
        const MatvecQ4Q8Args a{static_cast<const int*>(w.q), xq_, scratchScales(xq_, w.k / 32), y, w.dims, w.rows, w.k};
        if (two_rows_ && w.rows % 16 == 0) launch("matvec_q4x8_q8x2", &a, 8);  // min_chunk in threads (rows / 2)
        else launch("matvec_q4x8_q8", &a, 16);
    }
    void addRmsNormQuant(float* x, const float* t, const float* w) override {
        const int nb = m_.cfg.hidden / 32;
        const AddSumsqArgs s{x, t, xq_, nb};
        launch("add_sumsq", &s, 16);
        const NormQ8Args n{x, w, xq_, nb, m_.cfg.eps};
        launch("norm_q8x4", &n, 16);
    }
    void rope(float* x, int heads, int pos) override {
        const int pairs = m_.cfg.head_dim / 2;
        const RopeArgs a{x, pairs, pos, heads * pairs, std::log2(m_.cfg.rope_base)};
        launch("rope", &a, 0);
    }
    void appendKv(const float* k, const float* v, float* kt, float* vc, int pos) override {
        if (pos != ctx_ - 1) throw std::logic_error("appendKv: position must be the last of the context");
        const KvAppendArgs a{k, v, kt, vc, m_.attn_dims, m_.cfg.attnDims(ctx_)};
        launch("kv_append", &a, 0);
    }
    void attnScores(const float* q, const float* k, float* scores) override {
        const AttnScoresArgs a{q, k, scores, m_.attn_dims, m_.cfg.attnDims(ctx_),
                               1.0f / std::sqrt(static_cast<float>(m_.cfg.head_dim))};
        launch("attn_scores", &a, 64);
    }
    void softmax(float* scores) override {
        const SoftmaxArgs a{scores, ctx_, padTo8(ctx_), m_.cfg.heads};
        launch("softmax", &a, 1);
    }
    void attnValues(const float* p, const float* v, float* out) override {
        const AttnValuesArgs a{p, v, out, m_.attn_dims, m_.cfg.attnDims(ctx_)};
        launch("attn_values", &a, 64);
    }
    void siluMul(const float* g, const float* u, float* out, int n) override {
        const SiluMulArgs a{g, u, out, n};
        launch("silu_mul", &a, 1024);
    }
    void add(float* x, const float* t, int n) override {
        const pcuda::kernels::VectorAddArgs a{x, t, x, n};
        launch("vector_add", &a, 0);
    }
    void copy(float* dst, const float* src, int n) override {
        pcudaCheck(pcudaMemcpy(dst, src, size_t(n) * sizeof(float), pcuda::CopyKind::DeviceToDevice), "pcudaMemcpy");
    }
    void finish() override { pcudaCheck(pcudaDeviceSynchronize(), "pcudaDeviceSynchronize"); }
    void beginToken(bool profiling) override {
        abandonCapture();  // an earlier token threw before endToken
        if (!graphs_ || profiling) return;
        pcudaCheck(pcudaStreamBeginCapture(nullptr), "pcudaStreamBeginCapture");
        capturing_ = true;
    }
    void endToken() override {
        if (!capturing_) return;
        capturing_ = false;
        pcudaGraph_t g = nullptr;
        pcudaCheck(pcudaStreamEndCapture(nullptr, &g), "pcudaStreamEndCapture");
        const pcuda::Error e = pcudaGraphLaunch(g, nullptr);  // synchronous: logits ready after
        if (e != pcuda::Error::Success) {
            const std::string why = std::string("pcudaGraphLaunch: ") + pcudaGetErrorString(e) + " (" +
                                    pcudaGetLastErrorDetail() + ")";
            pcudaGraphDestroy(g);
            throw std::runtime_error(why);
        }
        pcudaCheck(pcudaGraphDestroy(g), "pcudaGraphDestroy");
    }

private:
    // min_chunk == 0: never chunked (too little work to share).
    void launch(const char* name, const void* args, int64_t min_chunk) {
        const pcuda::Error e = (workers_ && min_chunk) ? pcudaLaunchKernelChunked(name, args, workers_, min_chunk)
                                                        : pcudaLaunchKernel(name, args);
        if (e != pcuda::Error::Success) pcudaFail(name, e);
    }

    const Model& m_;
    unsigned     workers_;
    bool         graphs_;
    bool         two_rows_;
    bool         capturing_ = false;
    int          ctx_;
    int*         xq_ = nullptr;
};

class CppBackend final : public Backend {
public:
    explicit CppBackend(const Model& m) : m_(m), ctx_(m.cfg.max_ctx) {}

    void setScratch(int* q) override { xq_ = q; }
    void setContext(int ctx) override { ctx_ = ctx; }
    void matvec(const Mat& w, const float* x, float* y) override {
        if (w.format == Weights::Q4X8Q8) {
            ref::quantizeQ8x4(x, xq_, scratchScales(xq_, w.k / 32), w.k / 32);
            matvecQuantized(w, y);
        } else if (w.format == Weights::Q4X8) ref::matvecQ4x8(static_cast<const uint8_t*>(w.q), w.s, x, y, w.rows, w.k);
        else if (w.format == Weights::Q4) ref::matvecQ4(static_cast<const uint8_t*>(w.q), w.s, x, y, w.rows, w.k);
        else ref::matvecQ8(static_cast<const int8_t*>(w.q), w.s, x, y, w.rows, w.k);
    }
    void rmsNorm(const float* x, const float* w, float* y) override {
        ref::rmsNorm(x, w, y, m_.cfg.hidden, 1, m_.cfg.eps);
    }
    void addRmsNorm(float* x, const float* t, const float* w, float* y) override {
        ref::addRmsNorm(x, t, w, y, m_.cfg.hidden, m_.cfg.eps);
    }
    void siluMulQuant(const float* g, const float* u, int n) override {
        ref::siluMulQ8x4(g, u, xq_, scratchScales(xq_, n / 32), n / 32);
    }
    void addRmsNormQuant(float* x, const float* t, const float* w) override {
        ref::addSumsq(x, t, xq_, m_.cfg.hidden / 32);
        ref::normQ8x4(x, w, xq_, m_.cfg.hidden / 32, m_.cfg.eps);
    }
    void matvecQuantized(const Mat& w, float* y) override {
        ref::matvecQ4x8Q8(static_cast<const int*>(w.q), xq_, scratchScales(xq_, w.k / 32), y, w.rows, w.k);
    }
    void rope(float* x, int heads, int pos) override {
        const int pairs = m_.cfg.head_dim / 2;
        ref::rope(x, pairs, pos, heads * pairs, std::log2(m_.cfg.rope_base));
    }
    void appendKv(const float* k, const float* v, float* kt, float* vc, int pos) override {
        ref::appendKv(k, v, kt, vc, m_.cfg.kvStride(), m_.cfg.kStride(), pos);
    }
    void attnScores(const float* q, const float* k, float* scores) override {
        ref::attnScores(q, k, scores, m_.cfg.attnDims(ctx_), 1.0f / std::sqrt(static_cast<float>(m_.cfg.head_dim)));
    }
    void softmax(float* scores) override { ref::softmax(scores, ctx_, padTo8(ctx_), m_.cfg.heads); }
    void attnValues(const float* p, const float* v, float* out) override {
        ref::attnValues(p, v, out, m_.cfg.attnDims(ctx_));
    }
    void siluMul(const float* g, const float* u, float* out, int n) override { ref::siluMul(g, u, out, n); }
    void add(float* x, const float* t, int n) override { ref::add(x, t, n); }
    void copy(float* dst, const float* src, int n) override { std::memcpy(dst, src, size_t(n) * sizeof(float)); }
    void finish() override {}

private:
    const Model& m_;
    int          ctx_;
    int*         xq_ = nullptr;
};

} // namespace

std::unique_ptr<Backend> makePcudaBackend(const Model& m, unsigned workers, bool graphs, bool two_rows) {
    return std::make_unique<PcudaBackend>(m, workers, graphs, two_rows);
}

std::unique_ptr<Backend> makeCppBackend(const Model& m) { return std::make_unique<CppBackend>(m); }

} // namespace llm
