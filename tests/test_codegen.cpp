// Code generation paths: Booth's x86 object, native (Booth IR via --metal, compiled by MSVC)
// and simd (8 CUDA threads per AVX2 register) must give bit-identical results for every kernel,
// including partial 8-thread groups and chunked launches, and must contain faults the same way.

#include "harness.hpp"
#include "test_kernels.hpp"

#include "llm_kernels.hpp"

#include <pcuda/kernels.hpp>

#include <windows.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using pcuda::Codegen;
using pcuda::CopyKind;
using pcuda::Error;

namespace {

const Codegen kPaths[] = {Codegen::Booth, Codegen::Native, Codegen::Simd};

struct Fixture {
    RuntimeFixture rt;
    bool ok = false;
    Fixture() { ok = rt.ok && llm::registerKernels() == Error::Success; }
    ~Fixture() { pcudaSetKernelCodegen(Codegen::Auto); }
};

template <class T>
T* dev(const std::vector<T>& h) {
    void* p = nullptr;
    if (pcudaMalloc(&p, h.size() * sizeof(T)) != Error::Success) return nullptr;
    if (pcudaMemcpy(p, h.data(), h.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}

template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    pcudaMemcpy(h.data(), d, n * sizeof(T), CopyKind::DeviceToHost);
    return h;
}

std::vector<float> floats(size_t n, unsigned seed, float lo, float hi) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& f : v) f = d(g);
    return v;
}

std::vector<uint8_t> bytes(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = static_cast<uint8_t>(g());
    return v;
}

// Runs `launch` (which writes `out_bytes` at `out`) under every path, serial and chunked, and
// requires the bytes to match the Booth result exactly.
void sameOnAllPaths(const char* what, void* out, size_t out_bytes, const std::function<Error(bool chunked)>& launch,
                    bool chunkable) {
    std::vector<uint8_t> want;
    for (Codegen c : kPaths) {
        for (int chunked = 0; chunked <= (chunkable ? 1 : 0); ++chunked) {
            REQUIRE_OK(pcudaSetKernelCodegen(c));
            REQUIRE_OK(pcudaMemset(out, 0xA5, out_bytes));
            const Error e = launch(chunked != 0);
            if (e != Error::Success) {
                th::fail(__FILE__, __LINE__,
                         std::string(what) + " on " + pcudaCodegenName(c) + ": " + th::describe(e));
                return;
            }
            std::vector<uint8_t> got(out_bytes);
            REQUIRE_OK(pcudaMemcpy(got.data(), out, out_bytes, CopyKind::DeviceToHost));
            if (want.empty()) {
                want = got;
            } else if (got != want) {
                size_t i = 0;
                while (i < got.size() && got[i] == want[i]) ++i;
                th::fail(__FILE__, __LINE__,
                         std::string(what) + ": " + pcudaCodegenName(c) + (chunked ? " chunked" : "") +
                             " differs from booth at byte " + std::to_string(i));
                return;
            }
        }
    }
}

} // namespace

TEST_CASE(paths_resolve_as_documented) {
    Fixture fx;
    REQUIRE(fx.ok);
    Codegen used{};
    REQUIRE_OK(pcudaSetKernelCodegen(Codegen::Booth));
    REQUIRE_OK(pcudaGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Booth);
    REQUIRE_OK(pcudaSetKernelCodegen(Codegen::Native));
    REQUIRE_OK(pcudaGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Native);
    REQUIRE_OK(pcudaSetKernelCodegen(Codegen::Simd));
    REQUIRE_OK(pcudaGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Simd);
    // deep_recursion calls a helper function: no simd translation, native is used instead.
    REQUIRE_OK(pcudaGetKernelCodegenFor("deep_recursion", &used));
    CHECK(used == Codegen::Native);
    REQUIRE_OK(pcudaSetKernelCodegen(Codegen::Auto));
    REQUIRE_OK(pcudaGetKernelCodegenFor("matvec_q8", &used));
    CHECK(used == Codegen::Simd);
    CHECK_ERR(pcudaSetKernelCodegen(static_cast<Codegen>(9)), Error::InvalidValue);
}

TEST_CASE(elementwise_kernels_identical_on_all_paths) {
    Fixture fx;
    REQUIRE(fx.ok);
    for (int n : {1, 7, 8, 9, 63, 1000, 4097}) {
        const auto a = floats(size_t(n), 1, -4, 4), b = floats(size_t(n), 2, -4, 4);
        float *da = dev(a), *db = dev(b), *dout = dev(std::vector<float>(size_t(n)));
        REQUIRE(da && db && dout);
        const std::string tag = " n=" + std::to_string(n);
        sameOnAllPaths(("vector_add" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? pcuda::kernels::launchVectorAddChunked(da, db, dout, n, 4, 64)
                      : pcuda::kernels::launchVectorAdd(da, db, dout, n);
        }, true);
        const testkernels::ScaleAddArgs sa{da, dout, 2.5f, -1.25f, n};
        sameOnAllPaths(("scale_add" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? pcudaLaunchKernelChunked("scale_add", &sa, 4, 64) : pcudaLaunchKernel("scale_add", &sa);
        }, true);
        const testkernels::TrigMixArgs tm{da, dout, n};
        sameOnAllPaths(("trig_mix" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? pcudaLaunchKernelChunked("trig_mix", &tm, 4, 64) : pcudaLaunchKernel("trig_mix", &tm);
        }, true);
        const llm::SiluMulArgs si{da, db, dout, n};
        sameOnAllPaths(("silu_mul" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? pcudaLaunchKernelChunked("silu_mul", &si, 4, 64) : pcudaLaunchKernel("silu_mul", &si);
        }, true);
    }
    // Integer division (lane-by-lane in simd), negative operands included.
    std::vector<int> ia(1001);
    for (size_t i = 0; i < ia.size(); ++i) ia[i] = int(i * 7919 % 20001) - 10000;
    int *dia = dev(ia), *dio = dev(std::vector<int>(ia.size()));
    REQUIRE(dia && dio);
    const testkernels::IntDivArgs id{dia, dio, -7, int(ia.size())};
    sameOnAllPaths("int_div", dio, ia.size() * 4, [&](bool) { return pcudaLaunchKernel("int_div", &id); }, false);
}

TEST_CASE(llm_kernels_identical_on_all_paths) {
    Fixture fx;
    REQUIRE(fx.ok);
    // matvec_q8 and matvec_q4 with row counts that leave partial 8-row groups.
    for (auto [rows, k] : {std::pair{1, 32}, std::pair{13, 64}, std::pair{8, 32}, std::pair{1000, 256}, std::pair{1003, 256}}) {
        const auto q = bytes(size_t(rows) * k, 3);
        const auto s = floats(size_t(rows) * k / 32, 4, 0.001f, 0.02f);
        const auto x = floats(size_t(k), 5, -1, 1);
        auto* dq = dev(q);
        auto* ds = dev(s);
        auto* dx = dev(x);
        auto* dy = dev(std::vector<float>(size_t(rows)));
        auto* dd = dev(std::vector<int>{rows, k});
        REQUIRE(dq && ds && dx && dy && dd);
        const std::string tag = " " + std::to_string(rows) + "x" + std::to_string(k);
        const llm::MatvecQ8Args a8{reinterpret_cast<const int8_t*>(dq), ds, dx, dy, dd, rows, k};
        sameOnAllPaths(("matvec_q8" + tag).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
            return ch ? pcudaLaunchKernelChunked("matvec_q8", &a8, 4, 1) : pcudaLaunchKernel("matvec_q8", &a8);
        }, true);
        const llm::MatvecQ4Args a4{dq, ds, dx, dy, dd, rows, k};
        sameOnAllPaths(("matvec_q4" + tag).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
            return ch ? pcudaLaunchKernelChunked("matvec_q4", &a4, 4, 1) : pcudaLaunchKernel("matvec_q4", &a4);
        }, true);
        if (rows % 8 == 0)  // interleaved layout (the bytes are just a different arrangement here)
            sameOnAllPaths(("matvec_q4x8" + tag).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
                return ch ? pcudaLaunchKernelChunked("matvec_q4x8", &a4, 4, 8) : pcudaLaunchKernel("matvec_q4x8", &a4);
            }, true);
    }
    // Attention, softmax (fmax, varying loops), rms_norm (sqrt), rope (sin/cos).
    const llm::AttnDims d{6, 77, 64, 128, 3};
    const auto q = floats(size_t(d.n_heads) * d.head_dim, 6, -1, 1);
    const auto kv = floats(size_t(d.ctx) * d.kv_stride, 7, -1, 1);
    float *dq = dev(q), *dk = dev(kv), *dsc = dev(std::vector<float>(size_t(d.n_heads) * d.ctx)),
          *dout = dev(std::vector<float>(size_t(d.n_heads) * d.head_dim));
    int* dd = dev(std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group});
    REQUIRE(dq && dk && dsc && dout && dd);
    const llm::AttnScoresArgs sa{dq, dk, dsc, dd, d, 0.125f};
    sameOnAllPaths("attn_scores", dsc, size_t(d.n_heads) * d.ctx * 4, [&](bool ch) {
        return ch ? pcudaLaunchKernelChunked("attn_scores", &sa, 4, 64) : pcudaLaunchKernel("attn_scores", &sa);
    }, true);
    REQUIRE_OK(pcudaSetKernelCodegen(Codegen::Booth));
    REQUIRE_OK(pcudaLaunchKernel("attn_scores", &sa));
    const auto scores = host(dsc, size_t(d.n_heads) * d.ctx);
    const llm::SoftmaxArgs sm{dsc, d.ctx, d.n_heads};
    sameOnAllPaths("softmax", dsc, scores.size() * 4, [&](bool) {
        pcudaMemcpy(dsc, scores.data(), scores.size() * 4, CopyKind::HostToDevice);
        return pcudaLaunchKernel("softmax", &sm);
    }, false);
    const llm::AttnValuesArgs va{dsc, dk, dout, dd, d};
    sameOnAllPaths("attn_values", dout, size_t(d.n_heads) * d.head_dim * 4, [&](bool ch) {
        return ch ? pcudaLaunchKernelChunked("attn_values", &va, 4, 64) : pcudaLaunchKernel("attn_values", &va);
    }, true);

    const int n = 200, rows = 11;
    const auto xs = floats(size_t(n) * rows, 8, -3, 3), w = floats(size_t(n), 9, 0.5f, 1.5f);
    float *dxs = dev(xs), *dw = dev(w), *dys = dev(std::vector<float>(xs.size()));
    REQUIRE(dxs && dw && dys);
    const llm::RmsNormArgs rn{dxs, dw, dys, n, rows, 1e-5f};
    sameOnAllPaths("rms_norm", dys, xs.size() * 4, [&](bool ch) {
        return ch ? pcudaLaunchKernelChunked("rms_norm", &rn, 4, 1) : pcudaLaunchKernel("rms_norm", &rn);
    }, true);
    const llm::RopeArgs ro{dys, 50, 12345, rows * n / 2, 18.93156857f};
    sameOnAllPaths("rope", dys, xs.size() * 4, [&](bool) {
        pcudaMemcpy(dys, xs.data(), xs.size() * 4, CopyKind::HostToDevice);
        return pcudaLaunchKernel("rope", &ro);
    }, false);
}

TEST_CASE(faults_are_contained_on_every_path) {
    Fixture fx;
    REQUIRE(fx.ok);
    for (Codegen c : kPaths) {
        REQUIRE_OK(pcudaSetKernelCodegen(c));
        float* out = nullptr;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
        const testkernels::OobWriteArgs args{out, 1024};
        CHECK_ERR(pcudaLaunchKernel("oob_write", &args), Error::KernelFault);
        pcuda::FaultInfo f{};
        CHECK_OK(pcudaGetLastFault(&f));
        CHECK(f.code == EXCEPTION_ACCESS_VIOLATION);
        CHECK(f.access_type == 1);
        CHECK(f.in_kernel_image == (c == Codegen::Booth));
        REQUIRE_OK(pcudaDeviceReset());

        int *a = nullptr, *o = nullptr;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&o), 64 * sizeof(int)));
        const testkernels::IntDivArgs div0{a, o, 0, 64};
        CHECK_ERR(pcudaLaunchKernel("int_div", &div0), Error::KernelFault);
        CHECK_OK(pcudaGetLastFault(&f));
        CHECK(f.code == EXCEPTION_INT_DIVIDE_BY_ZERO);
        REQUIRE_OK(pcudaDeviceReset());
        REQUIRE(llm::registerKernels() == Error::Success);
    }
}
