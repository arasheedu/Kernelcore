// pcuda_llm_bench: token generation (decode) built from the same .cu kernels on every path:
//   booth   pcuda running Booth's own x86 code         (all CPU threads, chunked launches)
//   native  pcuda running Booth's IR compiled by MSVC  (Metal route, one thread per lane)
//   simd    pcuda, 8 CUDA threads per AVX2 register    (Metal route, warp-style)
//   cuda    the RTX GPU, kernels compiled by nvcc
//   c++     plain C++ reference, one thread (also the correctness oracle)
//
// Apples to apples with `ollama run llama3.2` (scripts/compare_llm.ps1 runs both):
//   same shapes (Llama 3.2 3B), same bytes of weights per token (Q4 here: 2.0 GB, Ollama's
//   Q4_K_M file: 2.02 GB), same prompt length (KV cache already filled), same number of
//   generated tokens, each attending over every position so far, logits back on the host.
//
//   pcuda_llm_bench [--model 3b|1b] [--weights q4x8q8|q4x8|q4|q8] [--prompt P] [--tokens T] [--budget S]
//                   [--workers N] [--paths booth,native,simd,cuda,cpp] [--graphs on|off] [--check]
// --graphs on (default): each token runs as one pcuda graph and one CUDA graph (launches recorded
//   and run as one batch). The per-op breakdown always runs launches one by one.
//
// Slow paths (booth, c++) time an evenly spaced subset of the T positions so a run stays within
// --budget seconds per path; the average per-token cost is over the same context range.
// --check: tiny model, correctness only (ctest); exit 77 when there is no CUDA device.

#include "cuda_kernels.hpp"
#include "cuda_llm.hpp"
#include "llm_model.hpp"

#include <pcuda/api.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using llm::Op;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---- CUDA backend -------------------------------------------------------------

class CudaAllocator final : public llm::Allocator {
public:
    ~CudaAllocator() override {
        for (void* p : blocks_) cudaFree(p);
    }
    void* alloc(size_t bytes) override {
        void* p = nullptr;
        ck(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc");
        blocks_.push_back(p);
        return p;
    }
    void upload(void* dst, const void* src, size_t bytes) override {
        ck(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    void download(void* dst, const void* src, size_t bytes) override {
        ck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }

private:
    std::vector<void*> blocks_;
};

class CudaBackend final : public llm::Backend {
public:
    CudaBackend(const llm::Model& m, cudaStream_t s) : m_(m), s_(s), ctx_(m.cfg.max_ctx) {}

    void setScratch(int* q) override { xq_ = q; }
    float* scales(int nb) const { return llm::scratchScales(xq_, nb); }

    void setContext(int ctx) override {
        ctx_ = ctx;
        const llm::AttnDims d = m_.cfg.attnDims(ctx);
        dims_ = {d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride};
        ck(cudaMemcpyAsync(m_.attn_dims, dims_.data(), sizeof dims_, cudaMemcpyHostToDevice, s_), "dims");
    }
    void matvec(const llm::Mat& w, const float* x, float* y) override {
        if (w.format == llm::Weights::Q4X8Q8) {
            ck(gpu::llm::quantizeQ8x4(x, xq_, scales(w.k / 32), w.k / 32, s_), "quantize_q8x4");
            matvecQuantized(w, y);
        } else if (w.format == llm::Weights::Q4X8)
            ck(gpu::llm::matvecQ4x8(static_cast<const uint8_t*>(w.q), w.s, x, y, w.dims, w.rows, s_), "matvec_q4x8");
        else if (w.format == llm::Weights::Q4)
            ck(gpu::llm::matvecQ4(static_cast<const uint8_t*>(w.q), w.s, x, y, w.dims, w.rows, s_), "matvec_q4");
        else
            ck(gpu::llm::matvecQ8(static_cast<const int8_t*>(w.q), w.s, x, y, w.dims, w.rows, s_), "matvec_q8");
    }
    void rmsNorm(const float* x, const float* w, float* y) override {
        ck(gpu::llm::rmsNorm(x, w, y, m_.cfg.hidden, 1, m_.cfg.eps, s_), "rms_norm");
    }
    void addRmsNorm(float* x, const float* t, const float* w, float* y) override {
        ck(gpu::llm::addRmsNorm(x, t, w, y, m_.cfg.hidden, m_.cfg.eps, s_), "add_rms_norm");
    }
    void siluMulQuant(const float* g, const float* u, int n) override {
        ck(gpu::llm::siluMulQ8x4(g, u, xq_, scales(n / 32), n / 32, s_), "silu_mul_q8x4");
    }
    void matvecQuantized(const llm::Mat& w, float* y) override {
        ck(gpu::llm::matvecQ4x8Q8(static_cast<const int*>(w.q), xq_, scales(w.k / 32), y, w.dims, w.rows, s_),
           "matvec_q4x8_q8");
    }
    void addRmsNormQuant(float* x, const float* t, const float* w) override {
        const int nb = m_.cfg.hidden / 32;
        ck(gpu::llm::addSumsq(x, t, xq_, nb, s_), "add_sumsq");
        ck(gpu::llm::normQ8x4(x, w, xq_, nb, m_.cfg.eps, s_), "norm_q8x4");
    }
    void rope(float* x, int heads, int pos) override {
        const int pairs = m_.cfg.head_dim / 2;
        ck(gpu::llm::rope(x, pairs, pos, heads * pairs, std::log2(m_.cfg.rope_base), s_), "rope");
    }
    void attnScores(const float* q, const float* k, float* scores) override {
        ck(gpu::llm::attnScores(q, k, scores, m_.attn_dims, m_.cfg.heads * llm::padTo8(ctx_),
                                1.0f / std::sqrt(static_cast<float>(m_.cfg.head_dim)), s_),
           "attn_scores");
    }
    void softmax(float* scores) override {
        ck(gpu::llm::softmax(scores, ctx_, llm::padTo8(ctx_), m_.cfg.heads, s_), "softmax");
    }
    void appendKv(const float* k, const float* v, float* kt, float* vc, int pos) override {
        if (pos != ctx_ - 1) throw std::logic_error("appendKv: position must be the last of the context");
        ck(gpu::llm::kvAppend(k, v, kt, vc, m_.attn_dims, m_.cfg.kvStride(), s_), "kv_append");
    }
    void attnValues(const float* p, const float* v, float* out) override {
        ck(gpu::llm::attnValues(p, v, out, m_.attn_dims, m_.cfg.heads * m_.cfg.head_dim, s_), "attn_values");
    }
    void siluMul(const float* g, const float* u, float* out, int n) override {
        ck(gpu::llm::siluMul(g, u, out, n, s_), "silu_mul");
    }
    void add(float* x, const float* t, int n) override { ck(gpu::vectorAdd(x, t, x, n, s_), "vector_add"); }
    void copy(float* dst, const float* src, int n) override {
        ck(cudaMemcpyAsync(dst, src, size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice, s_), "cudaMemcpy D2D");
    }
    void finish() override { ck(cudaStreamSynchronize(s_), "cudaStreamSynchronize"); }

    // CUDA graphs, as llama.cpp's CUDA backend uses them: capture the token's launches, update
    // the one executable graph in place (grid sizes and arguments change with the context),
    // re-instantiate only when the update is refused, and launch it.
    void useGraphs(bool on) { graphs_ = on; }
    void beginToken(bool profiling) override {
        if (!graphs_ || profiling) return;
        ck(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
        capturing_ = true;
    }
    void endToken() override {
        if (!capturing_) return;
        capturing_ = false;
        cudaGraph_t g = nullptr;
        ck(cudaStreamEndCapture(s_, &g), "cudaStreamEndCapture");
        cudaGraphExecUpdateResultInfo info{};
        if (!exec_ || cudaGraphExecUpdate(exec_, g, &info) != cudaSuccess) {
            cudaGetLastError();  // clear a refused update
            if (exec_) cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
            ck(cudaGraphInstantiate(&exec_, g, 0), "cudaGraphInstantiate");
        }
        ck(cudaGraphLaunch(exec_, s_), "cudaGraphLaunch");
        cudaGraphDestroy(g);
    }
    ~CudaBackend() override {
        if (exec_) cudaGraphExecDestroy(exec_);
    }

private:
    const llm::Model&  m_;
    cudaStream_t       s_;
    bool               graphs_ = false;
    bool               capturing_ = false;
    cudaGraphExec_t    exec_ = nullptr;
    int                ctx_;
    std::array<int, 7> dims_{};
    int*               xq_ = nullptr;
};

// ---- profilers ----------------------------------------------------------------

struct OpTimes {
    std::array<double, llm::kOpCount> ms{};
    double total() const {
        double t = 0;
        for (double v : ms) t += v;
        return t;
    }
};

class HostProfiler final : public llm::Profiler {
public:
    void begin(Op) override { start_ = Clock::now(); }
    void end(Op op) override { times.ms[size_t(op)] += msSince(start_); }
    OpTimes times;

private:
    Clock::time_point start_;
};

class CudaProfiler final : public llm::Profiler {
public:
    explicit CudaProfiler(cudaStream_t s) : s_(s) {}
    ~CudaProfiler() override {
        for (cudaEvent_t e : pool_) cudaEventDestroy(e);
    }
    void begin(Op op) override {
        cur_ = {op, event(), event()};
        ck(cudaEventRecord(cur_.a, s_), "cudaEventRecord");
    }
    void end(Op) override {
        ck(cudaEventRecord(cur_.b, s_), "cudaEventRecord");
        marks_.push_back(cur_);
    }
    OpTimes collect() {
        ck(cudaStreamSynchronize(s_), "cudaStreamSynchronize");
        OpTimes t;
        for (const Mark& m : marks_) {
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, m.a, m.b), "cudaEventElapsedTime");
            t.ms[size_t(m.op)] += ms;
        }
        marks_.clear();
        used_ = 0;
        return t;
    }

private:
    struct Mark {
        Op          op;
        cudaEvent_t a, b;
    };
    cudaEvent_t event() {
        if (used_ == pool_.size()) {
            cudaEvent_t e{};
            ck(cudaEventCreate(&e), "cudaEventCreate");
            pool_.push_back(e);
        }
        return pool_[used_++];
    }
    cudaStream_t             s_;
    std::vector<cudaEvent_t> pool_;
    size_t                   used_ = 0;
    std::vector<Mark>        marks_;
    Mark                     cur_{};
};

// ---- helpers ------------------------------------------------------------------

struct Agreement {
    double rel = 0;  // max |a - ref| / max |ref|
    bool   argmax = false;
    int    top5 = 0;
};

std::vector<int> topK(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = int(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[size_t(a)] > v[size_t(b)]; });
    idx.resize(size_t(k));
    return idx;
}

Agreement compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Agreement g;
    double m = 0, scale = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::isfinite(a[i]) ? std::abs(double(a[i]) - double(ref[i])) : 1e30;
        m = std::max(m, d);
        scale = std::max(scale, std::abs(double(ref[i])));
    }
    g.rel = scale > 0 ? m / scale : m;
    const auto ta = topK(a, 5), tr = topK(ref, 5);
    g.argmax = ta[0] == tr[0];
    for (int x : ta) g.top5 += int(std::count(tr.begin(), tr.end(), x));
    return g;
}

std::string fmtMs(double ms) {
    char b[32];
    if (ms >= 100) std::snprintf(b, sizeof b, "%.0f", ms);
    else if (ms >= 1) std::snprintf(b, sizeof b, "%.2f", ms);
    else std::snprintf(b, sizeof b, "%.3f", ms);
    return b;
}

// Positions to time: all T, or an evenly spaced subset when one token is slow.
std::vector<int> schedule(int prompt, int tokens, double est_ms, double budget_s) {
    int n = tokens;
    if (est_ms * tokens > budget_s * 1000.0) n = std::max(4, int(budget_s * 1000.0 / std::max(est_ms, 1e-3)));
    n = std::min(n, tokens);
    std::vector<int> pos;
    for (int i = 0; i < n; ++i) pos.push_back(prompt + int((i + 0.5) * tokens / n));
    return pos;
}

struct Column {
    std::string name;
    OpTimes     ops;           // one profiled token at the first generated position
    double      token_ms = 0;  // mean over the timed positions, unprofiled
    int         timed = 0;
};

// ---- main flow ----------------------------------------------------------------

int run(int argc, char** argv) {
    std::string model = "3b", weights = "q4x8q8";
    int prompt = 32, tokens = 128;
    double budget = 20;
    unsigned workers_opt = 0;
    bool check = false;
    std::string paths_opt = "booth,native,simd,cuda,cpp";
    bool graphs = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--check") check = true;
        else if (a == "--model") model = next();
        else if (a == "--weights") weights = next();
        else if (a == "--prompt") prompt = std::atoi(next().c_str());
        else if (a == "--tokens") tokens = std::atoi(next().c_str());
        else if (a == "--budget") budget = std::atof(next().c_str());
        else if (a == "--workers") workers_opt = static_cast<unsigned>(std::atoi(next().c_str()));
        else if (a == "--paths") paths_opt = next();
        else if (a == "--graphs") {
            const std::string v = next();
            if (v != "on" && v != "off") throw std::invalid_argument("--graphs takes on or off");
            graphs = v == "on";
        }
        else {
            std::fprintf(stderr, "usage: pcuda_llm_bench [--model 3b|1b] [--weights q4x8q8|q4x8|q4|q8] [--prompt P] "
                                 "[--tokens T] [--budget S] [--workers N] [--paths booth,native,simd,cuda,cpp] "
                                 "[--graphs on|off] [--check]\n");
            return 2;
        }
    }
    if ((model != "3b" && model != "1b") || (weights != "q4" && weights != "q4x8" && weights != "q4x8q8" && weights != "q8") || prompt < 0 || tokens < 1 ||
        prompt + tokens > 32768 || budget <= 0) {
        std::fprintf(stderr, "pcuda_llm_bench: bad arguments\n");
        return 2;
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("pcuda_llm_bench: no usable CUDA device; skipping\n");
        return 77;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    if (pcudaInit() != pcuda::Error::Success) throw std::runtime_error("pcudaInit failed");
    if (llm::registerKernels() != pcuda::Error::Success)
        throw std::runtime_error(std::string("registering LLM kernels: ") + pcudaGetLastErrorDetail());
    pcuda::DeviceProperties pprops{};
    if (pcudaGetDeviceProperties(&pprops, 0) != pcuda::Error::Success) throw std::runtime_error("pcuda device properties");
    // Default: one chunk share per thread that runs chunks (the pool plus the launching thread).
    const unsigned workers = workers_opt ? workers_opt : pprops.worker_count + 1;

    const llm::Weights wf = weights == "q4"       ? llm::Weights::Q4
                            : weights == "q4x8"   ? llm::Weights::Q4X8
                            : weights == "q4x8q8" ? llm::Weights::Q4X8Q8
                                                  : llm::Weights::Q8;
    if (check) prompt = 20, tokens = 6;
    const llm::Config cfg = check ? llm::tinyConfig(wf)
                                  : model == "1b" ? llm::llama32_1b(prompt + tokens, wf) : llm::llama32_3b(prompt + tokens, wf);
    const double gb = double(cfg.weightBytes()) / 1e9;
    std::printf("Model : %s, %d layers, hidden %d, FFN %d, %d heads (%d KV), head_dim %d, vocab %d\n", cfg.name.c_str(),
                cfg.layers, cfg.hidden, cfg.inter, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.vocab);
    std::printf("Weights: %.2f B parameters, %s = %.2f GB read per token\n", double(cfg.weightParams()) / 1e9,
                llm::weightsName(wf), gb);
    std::printf("Decode: prompt of %d tokens already in the KV cache, then %d generated tokens (positions %d..%d)\n",
                prompt, tokens, prompt, prompt + tokens - 1);
    std::printf("GPU   : %s, %zu MiB\n", prop.name, prop.totalGlobalMem >> 20);
    std::printf("CPU   : %u logical processors; pcuda pool of %u threads + the launching thread; workers = %u\n",
                std::thread::hardware_concurrency(), pprops.worker_count, workers);
    std::printf("Graphs: %s (each token as one pcuda graph and one CUDA graph)\n", graphs ? "on" : "off");
    std::fflush(stdout);

    cudaStream_t stream{};
    ck(cudaStreamCreate(&stream), "cudaStreamCreate");
    int rc = 0;
    {
        llm::PcudaAllocator pa;
        CudaAllocator ca;
        llm::HostAllocator ha;
        std::vector<llm::Model> models;
        const auto t0 = Clock::now();
        llm::buildModels(cfg, 2024, {&pa, &ca}, models);
        const llm::Model& pm = models[0];
        const llm::Model& cm = models[1];
        const llm::Acts pacts = llm::allocActs(cfg, pa, 99);
        const llm::Acts cacts = llm::allocActs(cfg, ca, 99);
        const llm::Acts hacts = llm::allocActs(cfg, ha, 99);
        float* host_logits = nullptr;
        ck(cudaMallocHost(reinterpret_cast<void**>(&host_logits), size_t(cfg.vocab) * sizeof(float)), "cudaMallocHost");
        std::printf("Built identical random weights for pcuda and the GPU in %.1f s\n\n", msSince(t0) / 1000.0);

        auto cpp = llm::makeCppBackend(pm);
        auto pcu = llm::makePcudaBackend(pm, workers, graphs);
        CudaBackend cuda(cm, stream);
        cuda.useGraphs(graphs);
        const int first = prompt;

        // ---- correctness: every path at the first generated position ----
        llm::decodeToken(*cpp, pm, hacts, first);
        const std::vector<float> ref(hacts.logits, hacts.logits + cfg.vocab);
        const pcuda::Codegen paths[] = {pcuda::Codegen::Booth, pcuda::Codegen::Native, pcuda::Codegen::Simd};
        std::vector<std::vector<float>> plog;
        for (pcuda::Codegen c : paths) {
            if (pcudaSetKernelCodegen(c) != pcuda::Error::Success) throw std::runtime_error("set codegen");
            llm::decodeToken(*pcu, pm, pacts, first);
            std::vector<float> l(size_t(cfg.vocab));
            pa.download(l.data(), pacts.logits, l.size() * sizeof(float));
            plog.push_back(std::move(l));
        }
        llm::decodeToken(cuda, cm, cacts, first);
        cuda.finish();
        std::vector<float> glog(size_t(cfg.vocab));
        ca.download(glog.data(), cacts.logits, glog.size() * sizeof(float));
        pcuda::Codegen mv{};
        pcudaGetKernelCodegenFor(wf == llm::Weights::Q4     ? "matvec_q4"
                                 : wf == llm::Weights::Q4X8 ? "matvec_q4x8"
                                 : wf == llm::Weights::Q4X8Q8 ? "matvec_q4x8_q8"
                                                              : "matvec_q8",
                                 &mv);

        const bool pcuda_same = plog[0] == plog[1] && plog[1] == plog[2];
        const Agreement ap = compare(plog[0], ref), ag = compare(glog, ref);
        std::printf("Correctness at position %d, logits vs the C++ reference (max |diff| / max |logit|):\n", first);
        std::printf("  pcuda booth / native / simd: %s; vs C++ %.2e, same argmax %s, top-5 %d/5\n",
                    pcuda_same ? "bit-identical to each other" : "DIFFER from each other", ap.rel,
                    ap.argmax ? "yes" : "NO", ap.top5);
        std::printf("  CUDA: %.2e, same argmax %s, top-5 %d/5 (nvcc contracts multiply-adds into FMA)\n\n", ag.rel,
                    ag.argmax ? "yes" : "NO", ag.top5);
        const bool ok = pcuda_same && ap.rel < 1e-4 && ap.argmax && ag.rel < 1e-2;

        if (!check) {
            // --paths: time only these. Each path heats the CPU and slows the paths timed after it.
            auto want = [&](const char* n) {
                return ("," + paths_opt + ",").find(std::string(",") + n + ",") != std::string::npos;
            };
            std::vector<Column> cols;
            auto timePcuda = [&](pcuda::Codegen c, const char* name) {
                pcudaSetKernelCodegen(c);
                Column col{name};
                llm::decodeToken(*pcu, pm, pacts, first);  // warm-up (pool, image load, caches)
                HostProfiler prof;
                llm::decodeToken(*pcu, pm, pacts, first, &prof);
                col.ops = prof.times;
                const auto pos = schedule(prompt, tokens, prof.times.total(), budget);
                const auto s = Clock::now();
                for (int p : pos) llm::decodeToken(*pcu, pm, pacts, p);  // logits are already host memory
                col.token_ms = msSince(s) / double(pos.size());
                col.timed = int(pos.size());
                std::printf("  timed %-13s %3d tokens: %8s ms/token\n", name, col.timed, fmtMs(col.token_ms).c_str());
                std::fflush(stdout);
                cols.push_back(col);
            };
            std::printf("Timing (slow paths sample evenly spaced positions within a %.0f s budget):\n", budget);
            if (want("booth")) timePcuda(pcuda::Codegen::Booth, "pcuda booth");
            if (want("native")) timePcuda(pcuda::Codegen::Native, "pcuda native");
            if (want("simd")) timePcuda(pcuda::Codegen::Simd, "pcuda simd");
            pcudaSetKernelCodegen(pcuda::Codegen::Auto);

            if (want("cuda")) {  // CUDA: warm to working clocks, profile one token, then all positions back to back
                const auto s = Clock::now();
                while (msSince(s) < 500) {
                    llm::decodeToken(cuda, cm, cacts, first);
                    cuda.finish();
                }
                Column col{"cuda"};
                CudaProfiler prof(stream);
                llm::decodeToken(cuda, cm, cacts, first, &prof);
                col.ops = prof.collect();
                const auto t = Clock::now();
                for (int p = prompt; p < prompt + tokens; ++p) {
                    llm::decodeToken(cuda, cm, cacts, p);
                    ck(cudaMemcpyAsync(host_logits, cacts.logits, size_t(cfg.vocab) * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "logits D2H");
                    cuda.finish();  // the host needs the logits before choosing the next token
                }
                col.token_ms = msSince(t) / tokens;
                col.timed = tokens;
                std::printf("  timed %-13s %3d tokens: %8s ms/token\n", "cuda", col.timed, fmtMs(col.token_ms).c_str());
                cols.push_back(col);
            }
            if (want("cpp")) {
                Column col{"c++ 1 thread"};
                HostProfiler prof;
                llm::decodeToken(*cpp, pm, hacts, first, &prof);
                col.ops = prof.times;
                const auto pos = schedule(prompt, tokens, prof.times.total(), budget);
                const auto s = Clock::now();
                for (int p : pos) llm::decodeToken(*cpp, pm, hacts, p);
                col.token_ms = msSince(s) / double(pos.size());
                col.timed = int(pos.size());
                std::printf("  timed %-13s %3d tokens: %8s ms/token\n", col.name.c_str(), col.timed,
                            fmtMs(col.token_ms).c_str());
                cols.push_back(col);
            }

            std::printf("\nTime per token, ms (breakdown: one profiled token at position %d)\n", first);
            std::printf("  %-34s", "operation");
            for (const Column& c : cols) std::printf(" %13s", c.name.c_str());
            std::printf("\n");
            for (int o = 0; o < llm::kOpCount; ++o) {
                std::printf("  %-34s", llm::opName(Op(o)));
                for (const Column& c : cols) std::printf(" %13s", fmtMs(c.ops.ms[size_t(o)]).c_str());
                std::printf("\n");
            }
            auto row = [&](const char* label, auto f) {
                std::printf("  %-34s", label);
                for (const Column& c : cols) std::printf(" %13s", f(c).c_str());
                std::printf("\n");
            };
            row("mean per token (unprofiled)", [](const Column& c) { return fmtMs(c.token_ms); });
            row("tokens per second", [](const Column& c) {
                char b[32];
                std::snprintf(b, sizeof b, "%.2f", 1000.0 / c.token_ms);
                return std::string(b);
            });
            row("weight bytes per second, GB/s", [&](const Column& c) {
                char b[32];
                std::snprintf(b, sizeof b, "%.1f", gb / (c.token_ms / 1000.0));
                return std::string(b);
            });
            row("tokens timed", [&](const Column& c) { return std::to_string(c.timed) + "/" + std::to_string(tokens); });
            std::printf("\n  matvec kernels ran on the '%s' path under auto.\n", pcudaCodegenName(mv));
            for (const Column& c : cols)
                std::printf("RESULT,%s,%.4f,%.4f,%.3f,%d\n", c.name.c_str(), c.token_ms, 1000.0 / c.token_ms,
                            gb / (c.token_ms / 1000.0), c.timed);
            std::printf("RESULT_INFO,%.4f,%d,%d\n", gb, prompt, tokens);
        }
        cudaFreeHost(host_logits);
        std::printf("\n%s\n", ok ? "PASS" : "FAIL");
        rc = ok ? 0 : 1;
    }
    cudaStreamDestroy(stream);
    pcudaShutdown();
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pcuda_llm_bench: %s\n", e.what());
        return 1;
    }
}
