// pcuda_kquant_vs_cuda: the K-quant kernels of the ggml backend (quantize_cols, matvec_q4k/q6k_q8,
// matmul_q4k/q6k_q8x4/x8) on pcuda (CPU) and on the NVIDIA GPU (the same .cu files, compiled by
// nvcc). Shapes: one Llama 3.2 3B layer as in a Q4_K_M file (q, k, o, gate, up Q4_K; v, down Q6_K),
// random weights. For each batch size it checks the GPU results against pcuda's and times the
// layer's matmuls (two input quantizations, seven matrices) on both.
//
//   pcuda_kquant_vs_cuda [--ncols 1,32,128,512] [--reps N] [--format per32|per256|both] [--check]
//
// --format: activation scales per 32 values (quantize_cols, matvec, matmul_*_q8x*) or per 256
// (quantize_cols_k, matmul_*_q8kx*: the ggml backend's batches; 1 token uses the 4-column kernel).
//
// Integer parts are exact on both. Floats may differ slightly: nvcc contracts a * b + c into FMA.
// --check: small shapes, correctness only (ctest); exit 77 when there is no CUDA device.

#include "cuda_llm.hpp"
#include "llm_kernels.hpp"
#include "llm_reference.hpp"

#include <pcuda/api.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void pk(pcuda::Error e, const char* what) {
    if (e != pcuda::Error::Success) throw std::runtime_error(std::string(what) + ": " + pcudaGetLastErrorDetail());
}

struct Matrix {
    const char* name;
    int         rows, k;
    bool        q6;
    int         input;  // 0: hidden-size activations, 1: FFN-size activations
};

// Device buffers of one matrix on both sides.
struct Weights {
    Matrix m;
    int*   pc = nullptr;  // pcuda
    int*   gp = nullptr;  // GPU
};

template <class T>
T* pcudaUpload(const std::vector<T>& v) {
    void* p = nullptr;
    pk(pcudaMalloc(&p, std::max<size_t>(1, v.size() * sizeof(T))), "pcudaMalloc");
    if (!v.empty()) pk(pcudaMemcpy(p, v.data(), v.size() * sizeof(T), pcuda::CopyKind::HostToDevice), "pcudaMemcpy");
    return static_cast<T*>(p);
}
template <class T>
T* gpuUpload(const std::vector<T>& v) {
    void* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(1, v.size() * sizeof(T))), "cudaMalloc");
    if (!v.empty()) ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return static_cast<T*>(p);
}
template <class T>
std::vector<T> pcudaDownload(const T* p, size_t n) {
    std::vector<T> v(n);
    pk(pcudaMemcpy(v.data(), p, n * sizeof(T), pcuda::CopyKind::DeviceToHost), "pcudaMemcpy");
    return v;
}
template <class T>
std::vector<T> gpuDownload(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return v;
}

// Random GGUF super-blocks with small positive fp16 scales, repacked as the backend does.
std::vector<int> randomWeights(const Matrix& m, unsigned seed) {
    const int nsb = m.k / 256, bb = m.q6 ? 210 : 144;
    std::mt19937 rng(seed);
    std::vector<uint8_t> src(size_t(m.rows) * nsb * bb);
    for (auto& v : src) v = static_cast<uint8_t>(rng());
    auto half = [&](uint8_t* p) {
        const uint16_t h = static_cast<uint16_t>(0x1800 + rng() % 0x0C00);
        p[0] = uint8_t(h & 0xFF), p[1] = uint8_t(h >> 8);
    };
    for (size_t r = 0; r < size_t(m.rows) * nsb; ++r) {
        uint8_t* b = src.data() + r * bb;
        if (m.q6) half(b + 208);
        else half(b), half(b + 2);
    }
    std::vector<int> qw(size_t(m.rows) * nsb * (m.q6 ? 53 : 37));
    if (m.q6) llm::ref::repackQ6K(src.data(), m.rows, m.k, qw.data());
    else llm::ref::repackQ4K(src.data(), m.rows, m.k, qw.data());
    return qw;
}

std::vector<int> parseList(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

struct Diff {
    size_t n = 0, identical = 0;
    double max_rel = 0;  // max |gpu - pcuda| / max |pcuda| over the output
};
Diff compare(const std::vector<float>& gpu, const std::vector<float>& pc) {
    Diff d;
    d.n = pc.size();
    double scale = 0;
    for (float v : pc) scale = std::max(scale, double(std::fabs(v)));
    for (size_t i = 0; i < pc.size(); ++i) {
        if (std::memcmp(&gpu[i], &pc[i], 4) == 0) ++d.identical;
        d.max_rel = std::max(d.max_rel, std::fabs(double(gpu[i]) - double(pc[i])) / (scale > 0 ? scale : 1));
    }
    return d;
}

} // namespace

int main(int argc, char** argv) try {
    std::vector<int> ncols_list = {1, 32, 128, 512};
    int  reps = 5;
    bool check = false;
    std::string format = "both";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ncols" && i + 1 < argc) ncols_list = parseList(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--check") check = true;
        else if (a == "--format" && i + 1 < argc) format = argv[++i];
        else {
            std::fprintf(stderr, "usage: pcuda_kquant_vs_cuda [--ncols 1,32,128,512] [--reps N] [--format per32|per256|both] [--check]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    pk(pcudaInit(), "pcudaInit");
    pk(llm::registerKernels(), "registerKernels");
    pcuda::DeviceProperties pp{};
    pk(pcudaGetDeviceProperties(&pp, 0), "pcudaGetDeviceProperties");
    const unsigned all_workers = pp.worker_count + 1, fast_workers = pp.performance_workers;

    // Llama 3.2 3B: hidden 3072, FFN 8192, 8 KV heads of 128, 28 layers. --check: small shapes.
    const int hidden = check ? 512 : 3072, ffn = check ? 768 : 8192, kv = check ? 256 : 1024, layers = 28;
    if (check) ncols_list = {1, 3, 5, 8, 13};
    const std::vector<Matrix> mats = {
        {"q", hidden, hidden, false, 0}, {"k", kv, hidden, false, 0},    {"v", kv, hidden, true, 0},
        {"o", hidden, hidden, false, 0}, {"gate", ffn, hidden, false, 0}, {"up", ffn, hidden, false, 0},
        {"down", hidden, ffn, true, 1},
    };
    std::printf("GPU  : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    std::printf("CPU  : pcuda pool of %u threads + caller (%u for bandwidth-bound launches)\n", pp.worker_count, fast_workers);
    std::printf("Layer: Llama 3.2 3B shapes, q/k/o/gate/up Q4_K, v/down Q6_K, random weights%s\n\n",
                check ? " (--check: small shapes)" : "");

    std::vector<Weights> ws;
    double weight_bytes = 0;
    for (size_t i = 0; i < mats.size(); ++i) {
        const auto qw = randomWeights(mats[i], unsigned(1234 + i));
        ws.push_back({mats[i], pcudaUpload(qw), gpuUpload(qw)});
        weight_bytes += double(qw.size()) * 4;
    }
    const int ins[2] = {hidden, ffn};

    bool ok = true;
    for (int q8k = 0; q8k < 2; ++q8k) {
    if ((q8k && format == "per32") || (!q8k && format == "per256")) continue;
    std::printf("Activations scaled per %s:\n", q8k ? "256 values (quantize_cols_k, matmul_*_q8kx*)" : "32 values (quantize_cols, matvec, matmul_*_q8x*)");
    std::printf("%6s | %-34s | %10s %10s | %12s %12s | %8s\n", "tokens", "GPU vs pcuda (all 7 matrices)", "pcuda ms", "GPU ms",
                "pcuda tok/s", "GPU tok/s", "GPU/pcuda");
    for (int ncols : ncols_list) {
        // Inputs and their 8-bit slots, per input size, on both sides.
        struct Input {
            int    k, nb;
            float *pc_x, *gp_x;
            int *  pc_q, *gp_q, *pc_dims, *gp_dims;
        } in[2];
        for (int j = 0; j < 2; ++j) {
            std::mt19937 rng(unsigned(77 + j + 10 * ncols));
            std::uniform_real_distribution<float> u(-3.0f, 3.0f);
            std::vector<float> x(size_t(ins[j]) * ncols);
            for (auto& v : x) v = u(rng);
            const int nb = ins[j] / 32;
            const int slot = q8k ? llm::scratchKInts(ins[j] / 256) : llm::scratchInts(nb);
            const std::vector<int> zero(size_t(slot) * ncols, 0), dims = {q8k ? ins[j] / 256 : nb, ncols};
            in[j] = {ins[j], nb, pcudaUpload(x), gpuUpload(x), pcudaUpload(zero), gpuUpload(zero), pcudaUpload(dims), gpuUpload(dims)};
        }
        std::vector<float*> pc_y, gp_y;
        std::vector<int*>   pc_md, gp_md;
        for (const auto& w : ws) {
            const std::vector<float> zero(size_t(w.m.rows) * ncols, 0.0f);
            const std::vector<int>   dims = {w.m.rows, w.m.k, ncols, 8};
            pc_y.push_back(pcudaUpload(zero)), gp_y.push_back(gpuUpload(zero));
            pc_md.push_back(pcudaUpload(dims)), gp_md.push_back(gpuUpload(dims));
        }
        const int tile = ncols > 4 ? 8 : 4;
        const char* const mm[2][2][2] = {{{"matmul_q4k_q8x4", "matmul_q6k_q8x4"}, {"matmul_q4k_q8x8", "matmul_q6k_q8x8"}},
                                         {{"matmul_q4k_q8kx4", "matmul_q6k_q8kx4"}, {"matmul_q4k_q8kx8", "matmul_q6k_q8kx8"}}};

        // One layer's matmuls, as the ggml backend launches them.
        auto run_pcuda = [&] {
            for (auto& x : in) {
                const llm::QuantizeColsArgs a{x.pc_x, x.pc_q, x.pc_dims, x.k, ncols};
                pk(pcudaLaunchKernelChunked(q8k ? "quantize_cols_k" : "quantize_cols", &a, fast_workers, q8k ? 8 : 16), "quantize");
            }
            for (size_t i = 0; i < ws.size(); ++i) {
                const auto& w = ws[i];
                const auto& x = in[w.m.input];
                if (ncols == 1 && !q8k) {
                    const llm::MatvecQ4Q8Args a{w.pc, x.pc_q, llm::scratchScales(x.pc_q, x.nb), pc_y[i], pc_md[i], w.m.rows, w.m.k};
                    pk(pcudaLaunchKernelChunked(w.m.q6 ? "matvec_q6k_q8" : "matvec_q4k_q8", &a, fast_workers, 16), "matvec");
                } else {
                    const llm::MatmulKQ8Args a{w.pc, x.pc_q, pc_y[i], pc_md[i], w.m.rows, w.m.k, ncols, 8};
                    pk(pcudaLaunchKernelChunked(mm[q8k][tile == 8][w.m.q6], &a, all_workers * 4, 16), "matmul");
                }
            }
        };
        auto run_gpu = [&] {
            for (auto& x : in)
                ck(q8k ? gpu::llm::quantizeColsK(x.gp_x, x.gp_q, x.gp_dims, x.k / 256, ncols, nullptr)
                       : gpu::llm::quantizeCols(x.gp_x, x.gp_q, x.gp_dims, x.nb, ncols, nullptr), "quantize");
            for (size_t i = 0; i < ws.size(); ++i) {
                const auto& w = ws[i];
                const auto& x = in[w.m.input];
                if (q8k)
                    ck(gpu::llm::matmulKQ8K(w.m.q6, tile, w.gp, x.gp_q, gp_y[i], gp_md[i], w.m.rows, ncols, nullptr), "matmul");
                else if (ncols == 1)
                    ck(gpu::llm::matvecKQ8(w.m.q6, w.gp, x.gp_q, llm::scratchScales(x.gp_q, x.nb), gp_y[i], gp_md[i], w.m.rows, nullptr), "matvec");
                else
                    ck(gpu::llm::matmulKQ8(w.m.q6, tile, w.gp, x.gp_q, gp_y[i], gp_md[i], w.m.rows, ncols, nullptr), "matmul");
            }
        };

        run_pcuda();
        run_gpu();
        ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // Correctness: the 8-bit activations must match exactly (per 32: words and {scale, sum}, the
        // partial-sum floats are not written; per 256: the whole slot); outputs closely.
        size_t qbad = 0;
        for (auto& x : in) {
            const int  slot = q8k ? llm::scratchKInts(x.k / 256) : llm::scratchInts(x.nb), used = q8k ? slot : x.nb * 10;
            const auto a = pcudaDownload(x.pc_q, size_t(slot) * ncols), b = gpuDownload(x.gp_q, size_t(slot) * ncols);
            for (int c = 0; c < ncols; ++c)
                for (int i = 0; i < used; ++i) qbad += a[size_t(c) * slot + i] != b[size_t(c) * slot + i];
        }
        Diff all;
        for (size_t i = 0; i < ws.size(); ++i) {
            const size_t n = size_t(ws[i].m.rows) * ncols;
            const Diff d = compare(gpuDownload(gp_y[i], n), pcudaDownload(pc_y[i], n));
            all.n += d.n, all.identical += d.identical, all.max_rel = std::max(all.max_rel, d.max_rel);
        }
        const bool good = qbad == 0 && all.max_rel < 1e-5;
        ok = ok && good;
        char verdict[64];
        std::snprintf(verdict, sizeof verdict, "%s %5.1f%% bit-equal, max %.1e%s", good ? "ok " : "BAD",
                      100.0 * double(all.identical) / double(all.n), all.max_rel, qbad ? " (q8 differs)" : "");

        // Timing: best of reps (after the warm-up above).
        double pc_ms = 1e30, gp_ms = 1e30;
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "cudaEventCreate");
        ck(cudaEventCreate(&e1), "cudaEventCreate");
        for (int r = 0; r < reps; ++r) {
            const auto t = Clock::now();
            run_pcuda();
            pc_ms = std::min(pc_ms, std::chrono::duration<double, std::milli>(Clock::now() - t).count());
            ck(cudaEventRecord(e0, nullptr), "cudaEventRecord");
            run_gpu();
            ck(cudaEventRecord(e1, nullptr), "cudaEventRecord");
            ck(cudaEventSynchronize(e1), "cudaEventSynchronize");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");
            gp_ms = std::min(gp_ms, double(ms));
        }
        cudaEventDestroy(e0), cudaEventDestroy(e1);
        // Matmul-only throughput for the whole model: 28 such layers per batch.
        const double pc_tps = ncols / (layers * pc_ms / 1000), gp_tps = ncols / (layers * gp_ms / 1000);
        std::printf("%6d | %-34s | %10.3f %10.3f | %12.1f %12.1f | %7.1fx\n", ncols, verdict, pc_ms, gp_ms, pc_tps, gp_tps,
                    pc_ms / gp_ms);

        for (auto& x : in) {
            pcudaFree(x.pc_x), pcudaFree(x.pc_q), pcudaFree(x.pc_dims);
            cudaFree(x.gp_x), cudaFree(x.gp_q), cudaFree(x.gp_dims);
        }
        for (size_t i = 0; i < ws.size(); ++i) pcudaFree(pc_y[i]), pcudaFree(pc_md[i]), cudaFree(gp_y[i]), cudaFree(gp_md[i]);
    }
    std::printf("\n");
    }
    std::printf("ms: one layer's 2 quantizations + 7 matmuls (%.1f MB of weights), best of %d.\n", weight_bytes / 1e6, reps);
    std::printf("tok/s: tokens / (28 layers x ms), matmuls only (no attention, norms or output layer).\n");
    for (auto& w : ws) pcudaFree(w.pc), cudaFree(w.gp);
    pcudaShutdown();
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "pcuda_kquant_vs_cuda: %s\n", e.what());
    return 1;
}
