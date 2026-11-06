// pcuda_vs_cuda: runs the same kernel sources on pcuda (Booth --cpu code on this CPU) and on
// the NVIDIA GPU (nvcc), compares the results and times both.
//
//   pcuda_vs_cuda           full table, 1K to 16M elements
//   pcuda_vs_cuda --quick   table up to 1M elements
//   pcuda_vs_cuda --check   correctness only, on small and awkward sizes; exit 1 on mismatch
//
// Exit code 77 means no usable CUDA device (ctest reports the check as skipped).

#include "cuda_kernels.hpp"
#include "test_kernels.hpp"

#include <pcuda/api.hpp>
#include <pcuda/kernels.hpp>

#include <cuda_runtime_api.h>
#include <intrin.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr float kScale = 2.5f;
constexpr float kBias = 1.0f;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

void ck(pcuda::Error e, const char* what) {
    if (e != pcuda::Error::Success)
        fail(std::string(what) + ": " + pcudaGetErrorString(e) + " (" + pcudaGetLastErrorDetail() + ")");
}

// ---- kernels under test -------------------------------------------------------

enum class Kind { VectorAdd, ScaleAdd, TrigMix };

struct Spec {
    Kind        kind;
    const char* name;
    const char* formula;
    int         inputs;
    float       lo, hi;       // input range (keeps results away from zero, so ULP is meaningful)
    int         max_ulp;      // allowed GPU vs pcuda difference
    double      max_rel;      // allowed error of either side vs the double reference
};

const Spec kSpecs[] = {
    {Kind::VectorAdd, "vector_add", "out[i] = a[i] + b[i]", 2, 1.0f, 2.0f, 0, 0.0},
    // nvcc contracts x*scale+bias to one FMA (one rounding); Booth rounds twice. Up to 1-2 ULP apart.
    {Kind::ScaleAdd, "scale_add", "out[i] = x[i] * 2.5f + 1.0f", 1, 0.0f, 1.0f, 2, 3.0e-7},
    // Different sinf/cosf/exp2f/log2f implementations (CUDA libdevice vs MSVC CRT + Booth's
    // argument lowering), so only a relative-error bound is meaningful.
    {Kind::TrigMix, "trig_mix", "out[i] = sinf(x)+cosf(x)+exp2f(x)+log2f(x+2)", 1, -1.0f, 1.0f, 64, 4.0e-6},
};

double reference(Kind k, float a, float b) {
    switch (k) {
    case Kind::VectorAdd: return double(a) + double(b);
    case Kind::ScaleAdd: return double(a) * kScale + kBias;
    case Kind::TrigMix: {
        const double x = a;
        return std::sin(x) + std::cos(x) + std::exp2(x) + std::log2(x + 2.0);
    }
    }
    return 0.0;
}

// Plain C++ compiled by cl /O2: what this CPU can do without Booth.
void cppKernel(Kind k, const float* __restrict a, const float* __restrict b, float* __restrict out, int n) {
    switch (k) {
    case Kind::VectorAdd:
        for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
        break;
    case Kind::ScaleAdd:
        for (int i = 0; i < n; ++i) out[i] = a[i] * kScale + kBias;
        break;
    case Kind::TrigMix:
        for (int i = 0; i < n; ++i)
            out[i] = std::sin(a[i]) + std::cos(a[i]) + std::exp2(a[i]) + std::log2(a[i] + 2.0f);
        break;
    }
}

// workers == 0: one synchronous launch on the calling thread; otherwise chunked over a pool.
void pcudaRun(Kind k, const float* a, const float* b, float* out, int n, unsigned workers) {
    switch (k) {
    case Kind::VectorAdd: {
        const pcuda::kernels::VectorAddArgs args{a, b, out, n};
        ck(workers ? pcudaLaunchKernelChunked("vector_add", &args, workers) : pcudaLaunchKernel("vector_add", &args),
           "pcuda vector_add");
        break;
    }
    case Kind::ScaleAdd: {
        const testkernels::ScaleAddArgs args{a, out, kScale, kBias, n};
        ck(workers ? pcudaLaunchKernelChunked("scale_add", &args, workers) : pcudaLaunchKernel("scale_add", &args),
           "pcuda scale_add");
        break;
    }
    case Kind::TrigMix: {
        const testkernels::TrigMixArgs args{a, out, n};
        ck(workers ? pcudaLaunchKernelChunked("trig_mix", &args, workers) : pcudaLaunchKernel("trig_mix", &args),
           "pcuda trig_mix");
        break;
    }
    }
}

void gpuLaunch(Kind k, const float* a, const float* b, float* out, int n, cudaStream_t s) {
    switch (k) {
    case Kind::VectorAdd: ck(gpu::vectorAdd(a, b, out, n, s), "cuda vector_add"); break;
    case Kind::ScaleAdd: ck(gpu::scaleAdd(a, out, kScale, kBias, n, s), "cuda scale_add"); break;
    case Kind::TrigMix: ck(gpu::trigMix(a, out, n, s), "cuda trig_mix"); break;
    }
}

// ---- buffers ------------------------------------------------------------------

struct Buffers {
    int    n = 0;
    float* h_in[2] = {};      // pinned host inputs (shared by every path)
    float* h_gpu = nullptr;   // pinned host copy of the GPU result
    std::vector<float> h_pcuda, h_cpp;
    float* p_in[2] = {};      // pcuda "device" memory
    float* p_out = nullptr;
    float* d_in[2] = {};      // CUDA device memory
    float* d_out = nullptr;

    explicit Buffers(int count) : n(count), h_pcuda(size_t(count)), h_cpp(size_t(count)) {
        const size_t bytes = size_t(n) * sizeof(float);
        for (int i = 0; i < 2; ++i) {
            ck(cudaMallocHost(reinterpret_cast<void**>(&h_in[i]), bytes), "cudaMallocHost");
            ck(pcudaMalloc(reinterpret_cast<void**>(&p_in[i]), bytes), "pcudaMalloc");
            ck(cudaMalloc(reinterpret_cast<void**>(&d_in[i]), bytes), "cudaMalloc");
        }
        ck(cudaMallocHost(reinterpret_cast<void**>(&h_gpu), bytes), "cudaMallocHost");
        ck(pcudaMalloc(reinterpret_cast<void**>(&p_out), bytes), "pcudaMalloc");
        ck(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes), "cudaMalloc");
    }
    ~Buffers() {
        for (int i = 0; i < 2; ++i) {
            cudaFreeHost(h_in[i]);
            pcudaFree(p_in[i]);
            cudaFree(d_in[i]);
        }
        cudaFreeHost(h_gpu);
        pcudaFree(p_out);
        cudaFree(d_out);
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

    size_t bytes() const { return size_t(n) * sizeof(float); }
};

void fillInputs(const Spec& s, Buffers& b, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(s.lo, s.hi);
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < b.n; ++i) b.h_in[j][i] = dist(rng);
}

// Copies inputs to both "devices" (pcuda inputs stay resident for the kernel-only timings).
void uploadInputs(const Spec& s, Buffers& b, cudaStream_t st) {
    for (int j = 0; j < s.inputs; ++j) {
        ck(pcudaMemcpy(b.p_in[j], b.h_in[j], b.bytes(), pcuda::CopyKind::HostToDevice), "pcudaMemcpy H2D");
        ck(cudaMemcpyAsync(b.d_in[j], b.h_in[j], b.bytes(), cudaMemcpyHostToDevice, st), "cudaMemcpy H2D");
    }
    ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
}

// ---- comparison ---------------------------------------------------------------

int64_t orderedBits(float f) {
    const int32_t i = std::bit_cast<int32_t>(f);
    return i >= 0 ? int64_t(i) : int64_t(INT32_MIN) - int64_t(i);
}

struct Accuracy {
    int64_t elements = 0;
    int64_t differ = 0;        // GPU vs pcuda, bitwise
    int64_t max_ulp = 0;       // GPU vs pcuda
    double  rel_pcuda = 0.0;   // max relative error vs double reference
    double  rel_gpu = 0.0;
    bool    nan = false;

    void add(const Spec& s, const Buffers& b) {
        for (int i = 0; i < b.n; ++i) {
            const float p = b.h_pcuda[size_t(i)], g = b.h_gpu[i];
            if (std::isnan(p) || std::isnan(g)) { nan = true; continue; }
            const double ref = reference(s.kind, b.h_in[0][i], b.h_in[1][i]);
            rel_pcuda = std::max(rel_pcuda, std::abs((double(p) - ref) / ref));
            rel_gpu = std::max(rel_gpu, std::abs((double(g) - ref) / ref));
            if (std::bit_cast<uint32_t>(p) != std::bit_cast<uint32_t>(g)) {
                ++differ;
                max_ulp = std::max(max_ulp, std::abs(orderedBits(p) - orderedBits(g)));
            }
        }
        elements += b.n;
    }

    // vector_add is exact on both sides: compare to the correctly rounded float sum.
    bool ok(const Spec& s) const {
        if (nan) return false;
        if (s.kind == Kind::VectorAdd) return differ == 0 && rel_pcuda <= 6.0e-8 && rel_gpu <= 6.0e-8;
        return max_ulp <= s.max_ulp && rel_pcuda <= s.max_rel && rel_gpu <= s.max_rel;
    }
};

// Runs each path once and fills h_pcuda / h_gpu. Also checks serial == chunked on pcuda.
void runOnceAndCollect(const Spec& s, Buffers& b, unsigned workers, cudaStream_t st, bool* chunk_mismatch) {
    ck(pcudaMemset(b.p_out, 0xFF, b.bytes()), "pcudaMemset");
    pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, b.n, 0);
    std::vector<float> serial(size_t(b.n));
    ck(pcudaMemcpy(serial.data(), b.p_out, b.bytes(), pcuda::CopyKind::DeviceToHost), "pcudaMemcpy D2H");

    ck(pcudaMemset(b.p_out, 0xFF, b.bytes()), "pcudaMemset");
    pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, b.n, workers);
    ck(pcudaMemcpy(b.h_pcuda.data(), b.p_out, b.bytes(), pcuda::CopyKind::DeviceToHost), "pcudaMemcpy D2H");
    *chunk_mismatch = std::memcmp(serial.data(), b.h_pcuda.data(), b.bytes()) != 0;

    ck(cudaMemsetAsync(b.d_out, 0xFF, b.bytes(), st), "cudaMemset");
    gpuLaunch(s.kind, b.d_in[0], b.d_in[1], b.d_out, b.n, st);
    ck(cudaMemcpyAsync(b.h_gpu, b.d_out, b.bytes(), cudaMemcpyDeviceToHost, st), "cudaMemcpy D2H");
    ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
}

// ---- timing -------------------------------------------------------------------

// Median of up to 21 samples after warm-up, capped at about one second per measurement
// (at least 3 samples) so the slow single-thread runs on 16M elements stay bearable.
template <class Sample>
double median(Sample&& sample) {
    const auto start = Clock::now();
    double first = sample();
    if (first < 100'000.0) sample();  // second warm-up unless one call is already slow
    std::vector<double> t;
    while (t.size() < 21) {
        t.push_back(sample());
        const double spent = std::chrono::duration<double>(Clock::now() - start).count();
        if (t.size() >= 3 && spent > 1.0) break;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <class F>
double hostMicros(F&& f) {
    const auto s = Clock::now();
    f();
    return std::chrono::duration<double, std::micro>(Clock::now() - s).count();
}

// Launches the kernel back to back for about 300 ms so the GPU reaches its working clocks.
void gpuWarmUp(const Spec& s, Buffers& b, cudaStream_t st) {
    const auto start = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - start).count() < 0.3) {
        for (int i = 0; i < 8; ++i) gpuLaunch(s.kind, b.d_in[0], b.d_in[1], b.d_out, b.n, st);
        ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
    }
}

struct Timings {
    double cpp, pcuda1, pcudaN, pcudaCopies, cudaKernel, cudaCopies;
};

Timings timeAll(const Spec& s, Buffers& b, unsigned workers, cudaStream_t st, cudaEvent_t e0, cudaEvent_t e1) {
    Timings t{};
    const int n = b.n;
    t.cpp = median([&] { return hostMicros([&] { cppKernel(s.kind, b.h_in[0], b.h_in[1], b.h_cpp.data(), n); }); });
    t.pcuda1 = median([&] { return hostMicros([&] { pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, n, 0); }); });
    t.pcudaN = median([&] { return hostMicros([&] { pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, n, workers); }); });
    // The same program shape a CUDA port runs: copy in, launch, copy out.
    t.pcudaCopies = median([&] {
        return hostMicros([&] {
            for (int j = 0; j < s.inputs; ++j)
                ck(pcudaMemcpy(b.p_in[j], b.h_in[j], b.bytes(), pcuda::CopyKind::HostToDevice), "pcudaMemcpy");
            pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, n, workers);
            ck(pcudaMemcpy(b.h_pcuda.data(), b.p_out, b.bytes(), pcuda::CopyKind::DeviceToHost), "pcudaMemcpy");
        });
    });
    // The GPU drops to idle clocks while the CPU measurements run (seconds at 16M elements),
    // and 21 short kernel samples finish before it ramps back up. Keep it busy first.
    gpuWarmUp(s, b, st);
    t.cudaKernel = median([&] {
        ck(cudaEventRecord(e0, st), "cudaEventRecord");
        gpuLaunch(s.kind, b.d_in[0], b.d_in[1], b.d_out, n, st);
        ck(cudaEventRecord(e1, st), "cudaEventRecord");
        ck(cudaEventSynchronize(e1), "cudaEventSynchronize");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");
        return double(ms) * 1000.0;
    });
    t.cudaCopies = median([&] {
        return hostMicros([&] {
            for (int j = 0; j < s.inputs; ++j)
                ck(cudaMemcpyAsync(b.d_in[j], b.h_in[j], b.bytes(), cudaMemcpyHostToDevice, st), "cudaMemcpy");
            gpuLaunch(s.kind, b.d_in[0], b.d_in[1], b.d_out, n, st);
            ck(cudaMemcpyAsync(b.h_gpu, b.d_out, b.bytes(), cudaMemcpyDeviceToHost, st), "cudaMemcpy");
            ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        });
    });
    return t;
}

// ---- reporting ----------------------------------------------------------------

std::string cpuName() {
    int r[12] = {};
    __cpuid(r, 0x80000002);
    __cpuid(r + 4, 0x80000003);
    __cpuid(r + 8, 0x80000004);
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    name.erase(0, name.find_first_not_of(' '));
    name.erase(name.find_last_not_of(' ') + 1);
    return name;
}

std::string fmtTime(double us) {
    char buf[32];
    if (us >= 100'000.0) std::snprintf(buf, sizeof buf, "%.0f ms", us / 1000.0);
    else if (us >= 1000.0) std::snprintf(buf, sizeof buf, "%.2f ms", us / 1000.0);
    else std::snprintf(buf, sizeof buf, "%.1f us", us);
    return buf;
}

std::string fmtRatio(double r) {
    char buf[32];
    if (r >= 10.0) std::snprintf(buf, sizeof buf, "%.0fx", r);
    else std::snprintf(buf, sizeof buf, "%.2fx", r);
    return buf;
}

std::string fmtCount(int n) {
    char buf[32];
    if (n >= (1 << 20) && n % (1 << 20) == 0) std::snprintf(buf, sizeof buf, "%dM", n >> 20);
    else if (n >= 1024 && n % 1024 == 0) std::snprintf(buf, sizeof buf, "%dK", n >> 10);
    else std::snprintf(buf, sizeof buf, "%d", n);
    return buf;
}

void printAccuracy(const Spec& s, const Accuracy& a, bool chunk_mismatch) {
    std::printf("  accuracy over %lld elements: GPU vs pcuda %lld differ (max %lld ULP); "
                "max relative error vs double: pcuda %.2e, GPU %.2e%s%s\n",
                static_cast<long long>(a.elements), static_cast<long long>(a.differ),
                static_cast<long long>(a.max_ulp), a.rel_pcuda, a.rel_gpu, a.nan ? "; NaN seen" : "",
                chunk_mismatch ? "; pcuda chunked != serial" : "");
    std::printf("  %s\n", a.ok(s) && !chunk_mismatch ? "PASS" : "FAIL");
}

int launchOverhead(cudaStream_t st, unsigned workers) {
    Buffers b(64);
    const Spec& s = kSpecs[0];
    fillInputs(s, b, 1);
    uploadInputs(s, b, st);
    const double p1 = median([&] { return hostMicros([&] { pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, 1, 0); }); });
    const double pN =
        median([&] { return hostMicros([&] { pcudaRun(s.kind, b.p_in[0], b.p_in[1], b.p_out, 1, workers); }); });
    const double g = median([&] {
        return hostMicros([&] {
            gpuLaunch(s.kind, b.d_in[0], b.d_in[1], b.d_out, 1, st);
            ck(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        });
    });
    std::printf("\nLaunch overhead (1 element, launch and wait): pcuda %s, pcuda chunked %s, CUDA %s\n",
                fmtTime(p1).c_str(), fmtTime(pN).c_str(), fmtTime(g).c_str());
    return 0;
}

int run(int argc, char** argv) {
    bool check = false, quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--quick")) quick = true;
        else {
            std::fprintf(stderr, "usage: pcuda_vs_cuda [--check | --quick]\n");
            return 2;
        }
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("pcuda_vs_cuda: no usable CUDA device (%s); skipping\n", cudaGetErrorString(cudaGetLastError()));
        return 77;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);

    ck(pcudaInit(), "pcudaInit");
    ck(testkernels::registerAll(), "register test kernels");
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());

    std::printf("GPU : %s, compute %d.%d, %d SMs, %zu MiB (driver %d.%d, runtime %d.%d)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount, prop.totalGlobalMem >> 20, drv / 1000, (drv % 1000) / 10,
                rt / 1000, (rt % 1000) / 10);
    std::printf("CPU : %s, %u logical processors (pcuda chunked launches use all of them)\n", cpuName().c_str(),
                workers);
    std::printf("Same .cu source: Booth kath --cpu for pcuda, nvcc for the GPU.\n");

    cudaStream_t st{};
    cudaEvent_t e0{}, e1{};
    ck(cudaStreamCreate(&st), "cudaStreamCreate");
    ck(cudaEventCreate(&e0), "cudaEventCreate");
    ck(cudaEventCreate(&e1), "cudaEventCreate");

    std::vector<int> sizes;
    if (check) sizes = {1, 7, 255, 256, 257, 1000, 4097, 65537, (1 << 20) + 3};
    else if (quick) sizes = {1 << 10, 1 << 14, 1 << 18, 1 << 20};
    else sizes = {1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24};

    if (!check)
        std::printf("Times are medians (up to 21 runs after warm-up). Ratios: how many times faster the GPU is.\n");

    bool all_ok = true;
    for (const Spec& s : kSpecs) {
        std::printf("\n%s: %s\n", s.name, s.formula);
        if (!check)
            std::printf("  %9s %10s %10s %10s %13s %12s %13s %9s %9s\n", "elements", "C++ /O2", "pcuda 1T",
                        ("pcuda " + std::to_string(workers) + "T").c_str(), "pcuda+copies", "CUDA kernel",
                        "CUDA+copies", "kernel", "end2end");
        Accuracy acc;
        bool chunk_mismatch = false;
        for (int n : sizes) {
            Buffers b(n);
            fillInputs(s, b, 12345u + unsigned(n));
            uploadInputs(s, b, st);
            bool mm = false;
            runOnceAndCollect(s, b, workers, st, &mm);
            chunk_mismatch |= mm;
            acc.add(s, b);
            if (check) continue;
            const Timings t = timeAll(s, b, workers, st, e0, e1);
            std::printf("  %9s %10s %10s %10s %13s %12s %13s %9s %9s\n", fmtCount(n).c_str(), fmtTime(t.cpp).c_str(),
                        fmtTime(t.pcuda1).c_str(), fmtTime(t.pcudaN).c_str(), fmtTime(t.pcudaCopies).c_str(),
                        fmtTime(t.cudaKernel).c_str(), fmtTime(t.cudaCopies).c_str(),
                        fmtRatio(std::min(t.pcuda1, t.pcudaN) / t.cudaKernel).c_str(),
                        fmtRatio(t.pcudaCopies / t.cudaCopies).c_str());
            std::fflush(stdout);
        }
        printAccuracy(s, acc, chunk_mismatch);
        all_ok &= acc.ok(s) && !chunk_mismatch;
    }

    if (!check) {
        launchOverhead(st, workers);
        std::printf("\nkernel  = best pcuda (1T or %uT) / CUDA kernel time (cudaEvent).\n"
                    "end2end = pcuda+copies / CUDA+copies (copy inputs in, launch, copy result out; pinned host memory).\n",
                    workers);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);
    pcudaShutdown();
    std::printf("\n%s\n", all_ok ? "ALL PASS" : "SOME CHECKS FAILED");
    return all_ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pcuda_vs_cuda: %s\n", e.what());
        return 1;
    }
}
