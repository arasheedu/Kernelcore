// Code generation paths: Booth's x86 object, native (Booth IR via --metal, compiled by MSVC)
// and simd (8 CUDA threads per AVX2 register) must give bit-identical results for every kernel,
// including partial 8-thread groups and chunked launches, and must contain faults the same way.

#include "harness.hpp"
#include "test_kernels.hpp"

#include "llm_kernels.hpp"

#include <kernelcore/kernels.hpp>

#include <windows.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using kc::Codegen;
using kc::CopyKind;
using kc::Error;

namespace {

const Codegen kPaths[] = {Codegen::Booth, Codegen::Native, Codegen::Simd};

struct Fixture {
    RuntimeFixture rt;
    bool ok = false;
    Fixture() { ok = rt.ok && llm::registerKernels() == Error::Success; }
    ~Fixture() { kcSetKernelCodegen(Codegen::Auto); }
};

template <class T>
T* dev(const std::vector<T>& h) {
    void* p = nullptr;
    if (kcMalloc(&p, h.size() * sizeof(T)) != Error::Success) return nullptr;
    if (kcMemcpy(p, h.data(), h.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}

template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    kcMemcpy(h.data(), d, n * sizeof(T), CopyKind::DeviceToHost);
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
            REQUIRE_OK(kcSetKernelCodegen(c));
            REQUIRE_OK(kcMemset(out, 0xA5, out_bytes));
            const Error e = launch(chunked != 0);
            if (e != Error::Success) {
                th::fail(__FILE__, __LINE__,
                         std::string(what) + " on " + kcCodegenName(c) + ": " + th::describe(e));
                return;
            }
            std::vector<uint8_t> got(out_bytes);
            REQUIRE_OK(kcMemcpy(got.data(), out, out_bytes, CopyKind::DeviceToHost));
            if (want.empty()) {
                want = got;
            } else if (got != want) {
                size_t i = 0;
                while (i < got.size() && got[i] == want[i]) ++i;
                th::fail(__FILE__, __LINE__,
                         std::string(what) + ": " + kcCodegenName(c) + (chunked ? " chunked" : "") +
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
    REQUIRE_OK(kcSetKernelCodegen(Codegen::Booth));
    REQUIRE_OK(kcGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Booth);
    REQUIRE_OK(kcSetKernelCodegen(Codegen::Native));
    REQUIRE_OK(kcGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Native);
    REQUIRE_OK(kcSetKernelCodegen(Codegen::Simd));
    REQUIRE_OK(kcGetKernelCodegenFor("vector_add", &used));
    CHECK(used == Codegen::Simd);
    // deep_recursion calls a helper function: no simd translation, native is used instead.
    REQUIRE_OK(kcGetKernelCodegenFor("deep_recursion", &used));
    CHECK(used == Codegen::Native);
    REQUIRE_OK(kcSetKernelCodegen(Codegen::Auto));
    REQUIRE_OK(kcGetKernelCodegenFor("matvec_q8", &used));
    CHECK(used == Codegen::Simd);
    CHECK_ERR(kcSetKernelCodegen(static_cast<Codegen>(9)), Error::InvalidValue);
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
            return ch ? kc::kernels::launchVectorAddChunked(da, db, dout, n, 4, 64)
                      : kc::kernels::launchVectorAdd(da, db, dout, n);
        }, true);
        const testkernels::ScaleAddArgs sa{da, dout, 2.5f, -1.25f, n};
        sameOnAllPaths(("scale_add" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("scale_add", &sa, 4, 64) : kcLaunchKernel("scale_add", &sa);
        }, true);
        const testkernels::TrigMixArgs tm{da, dout, n};
        sameOnAllPaths(("trig_mix" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("trig_mix", &tm, 4, 64) : kcLaunchKernel("trig_mix", &tm);
        }, true);
        const llm::SiluMulArgs si{da, db, dout, n};
        sameOnAllPaths(("silu_mul" + tag).c_str(), dout, size_t(n) * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("silu_mul", &si, 4, 64) : kcLaunchKernel("silu_mul", &si);
        }, true);
    }
    // Integer division (lane-by-lane in simd), negative operands included.
    std::vector<int> ia(1001);
    for (size_t i = 0; i < ia.size(); ++i) ia[i] = int(i * 7919 % 20001) - 10000;
    int *dia = dev(ia), *dio = dev(std::vector<int>(ia.size()));
    REQUIRE(dia && dio);
    const testkernels::IntDivArgs id{dia, dio, -7, int(ia.size())};
    sameOnAllPaths("int_div", dio, ia.size() * 4, [&](bool) { return kcLaunchKernel("int_div", &id); }, false);
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
            return ch ? kcLaunchKernelChunked("matvec_q8", &a8, 4, 1) : kcLaunchKernel("matvec_q8", &a8);
        }, true);
        const llm::MatvecQ4Args a4{dq, ds, dx, dy, dd, rows, k};
        sameOnAllPaths(("matvec_q4" + tag).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("matvec_q4", &a4, 4, 1) : kcLaunchKernel("matvec_q4", &a4);
        }, true);
        if (rows % 8 == 0)  // interleaved layout (the bytes are just a different arrangement here)
            sameOnAllPaths(("matvec_q4x8" + tag).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
                return ch ? kcLaunchKernelChunked("matvec_q4x8", &a4, 4, 8) : kcLaunchKernel("matvec_q4x8", &a4);
            }, true);
    }
    // 8-bit activations and the integer dot-product matvec (AVX-VNNI on the simd path).
    for (auto [rows, k] : {std::pair{8, 32}, std::pair{1000, 256}, std::pair{1008, 96}}) {
        const auto x = floats(size_t(k), 11, -2, 2);
        const auto w = bytes(size_t(rows) * k * 5 / 8, 12);  // any bit pattern is a valid stream
        std::vector<int> qw(w.size() / 4);
        std::memcpy(qw.data(), w.data(), w.size());
        for (size_t i = 0; i < qw.size(); ++i)  // keep the in-stream scales finite
            if (i % 40 >= 32) qw[i] = std::bit_cast<int>(0.004f + float(i % 7) * 0.001f);
        const int nb = k / 32;
        float* dx = dev(x);
        int* dqw = dev(qw);
        int* dxq = dev(std::vector<int>(size_t(nb) * 8));
        float* dxd = dev(std::vector<float>(size_t(nb) * 2));
        float* dy = dev(std::vector<float>(size_t(rows)));
        int* dd = dev(std::vector<int>{rows, k});
        REQUIRE(dx && dqw && dxq && dxd && dy && dd);
        const llm::QuantizeArgs qa{dx, dxq, dxd, nb};
        sameOnAllPaths("quantize_q8x4", dxq, size_t(nb) * 8 * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("quantize_q8x4", &qa, 4, 1) : kcLaunchKernel("quantize_q8x4", &qa);
        }, true);
        REQUIRE_OK(kcSetKernelCodegen(Codegen::Booth));
        REQUIRE_OK(kcLaunchKernel("quantize_q8x4", &qa));
        const llm::MatvecQ4Q8Args a{dqw, dxq, dxd, dy, dd, rows, k};
        sameOnAllPaths(("matvec_q4x8_q8 " + std::to_string(rows)).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("matvec_q4x8_q8", &a, 4, 8) : kcLaunchKernel("matvec_q4x8_q8", &a);
        }, true);
        if (rows % 16 == 0) {
            const auto one_row = host(dy, size_t(rows));  // the last path's matvec_q4x8_q8 result
            sameOnAllPaths(("matvec_q4x8_q8x2 " + std::to_string(rows)).c_str(), dy, size_t(rows) * 4, [&](bool ch) {
                return ch ? kcLaunchKernelChunked("matvec_q4x8_q8x2", &a, 4, 8) : kcLaunchKernel("matvec_q4x8_q8x2", &a);
            }, true);
            CHECK(host(dy, size_t(rows)) == one_row);
        }
    }

    {  // parallel residual add + RMS norm + quantize across paths (chunked too)
        const int nb = 37;
        const auto x = floats(size_t(nb) * 32, 15, -2, 2), t = floats(size_t(nb) * 32, 16, -2, 2),
                   w = floats(size_t(nb) * 32, 17, 0.9f, 1.1f);
        float *dx = dev(x), *dt = dev(t), *dw = dev(w);
        int* dq = dev(std::vector<int>(size_t(llm::scratchInts(nb))));
        REQUIRE(dx && dt && dw && dq);
        const llm::AddSumsqArgs s{dx, dt, dq, nb};
        const llm::NormQ8Args n{dx, dw, dq, nb, 1e-5f};
        sameOnAllPaths("add_sumsq + norm_q8x4", dq, size_t(llm::scratchInts(nb)) * 4, [&](bool ch) {
            kcMemcpy(dx, x.data(), x.size() * 4, CopyKind::HostToDevice);  // x is updated in place
            const Error e = ch ? kcLaunchKernelChunked("add_sumsq", &s, 4, 16) : kcLaunchKernel("add_sumsq", &s);
            if (e != Error::Success) return e;
            return ch ? kcLaunchKernelChunked("norm_q8x4", &n, 4, 16) : kcLaunchKernel("norm_q8x4", &n);
        }, true);
    }

    {  // fused SwiGLU + quantize across paths (chunked too)
        const int count = 77;
        const auto g = floats(size_t(count) * 32, 13, -6, 6), u = floats(size_t(count) * 32, 14, -2, 2);
        float *dg = dev(g), *du = dev(u);
        int* dxq = dev(std::vector<int>(size_t(count) * 8));
        float* dxd = dev(std::vector<float>(size_t(count) * 2));
        REQUIRE(dg && du && dxq && dxd);
        const llm::SiluMulQ8Args sq{dg, du, dxq, dxd, count};
        sameOnAllPaths("silu_mul_q8x4", dxq, size_t(count) * 8 * 4, [&](bool ch) {
            return ch ? kcLaunchKernelChunked("silu_mul_q8x4", &sq, 4, 8) : kcLaunchKernel("silu_mul_q8x4", &sq);
        }, true);
    }

    // Attention, softmax (fmax, varying loops), rms_norm (sqrt), rope (sin/cos).
    // ctx_pad 80, k_stride 77: K (transposed, 128 x 77) and V (77 x 128) share one buffer.
    const llm::AttnDims d{6, 77, 64, 128, 3, 80, 77};
    const auto q = floats(size_t(d.n_heads) * d.head_dim, 6, -1, 1);
    const auto kv = floats(size_t(d.ctx) * d.kv_stride, 7, -1, 1);
    float *dq = dev(q), *dk = dev(kv), *dsc = dev(std::vector<float>(size_t(d.n_heads) * d.ctx_pad)),
          *dout = dev(std::vector<float>(size_t(d.n_heads) * d.head_dim));
    int* dd = dev(std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride});
    REQUIRE(dq && dk && dsc && dout && dd);
    const llm::AttnScoresArgs sa{dq, dk, dsc, dd, d, 0.125f};
    sameOnAllPaths("attn_scores", dsc, size_t(d.n_heads) * d.ctx_pad * 4, [&](bool ch) {
        return ch ? kcLaunchKernelChunked("attn_scores", &sa, 4, 64) : kcLaunchKernel("attn_scores", &sa);
    }, true);
    REQUIRE_OK(kcSetKernelCodegen(Codegen::Booth));
    REQUIRE_OK(kcLaunchKernel("attn_scores", &sa));
    const auto scores = host(dsc, size_t(d.n_heads) * d.ctx_pad);
    const llm::SoftmaxArgs sm{dsc, d.ctx, d.ctx_pad, d.n_heads};
    sameOnAllPaths("softmax", dsc, scores.size() * 4, [&](bool) {
        kcMemcpy(dsc, scores.data(), scores.size() * 4, CopyKind::HostToDevice);
        return kcLaunchKernel("softmax", &sm);
    }, false);
    const llm::AttnValuesArgs va{dsc, dk, dout, dd, d};
    sameOnAllPaths("attn_values", dout, size_t(d.n_heads) * d.head_dim * 4, [&](bool ch) {
        return ch ? kcLaunchKernelChunked("attn_values", &va, 4, 64) : kcLaunchKernel("attn_values", &va);
    }, true);

    const int n = 200, rows = 11;
    const auto xs = floats(size_t(n) * rows, 8, -3, 3), w = floats(size_t(n), 9, 0.5f, 1.5f);
    float *dxs = dev(xs), *dw = dev(w), *dys = dev(std::vector<float>(xs.size()));
    REQUIRE(dxs && dw && dys);
    const llm::RmsNormArgs rn{dxs, dw, dys, n, rows, 1e-5f};
    sameOnAllPaths("rms_norm", dys, xs.size() * 4, [&](bool ch) {
        return ch ? kcLaunchKernelChunked("rms_norm", &rn, 4, 1) : kcLaunchKernel("rms_norm", &rn);
    }, true);
    const llm::RopeArgs ro{dys, 50, 12345, rows * n / 2, 18.93156857f};
    sameOnAllPaths("rope", dys, xs.size() * 4, [&](bool) {
        kcMemcpy(dys, xs.data(), xs.size() * 4, CopyKind::HostToDevice);
        return kcLaunchKernel("rope", &ro);
    }, false);
}

TEST_CASE(faults_are_contained_on_every_path) {
    Fixture fx;
    REQUIRE(fx.ok);
    for (Codegen c : kPaths) {
        REQUIRE_OK(kcSetKernelCodegen(c));
        float* out = nullptr;
        REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
        const testkernels::OobWriteArgs args{out, 1024};
        CHECK_ERR(kcLaunchKernel("oob_write", &args), Error::KernelFault);
        kc::FaultInfo f{};
        CHECK_OK(kcGetLastFault(&f));
        CHECK(f.code == EXCEPTION_ACCESS_VIOLATION);
        CHECK(f.access_type == 1);
        CHECK(f.in_kernel_image == (c == Codegen::Booth));
        REQUIRE_OK(kcDeviceReset());

        int *a = nullptr, *o = nullptr;
        REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
        REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&o), 64 * sizeof(int)));
        const testkernels::IntDivArgs div0{a, o, 0, 64};
        CHECK_ERR(kcLaunchKernel("int_div", &div0), Error::KernelFault);
        CHECK_OK(kcGetLastFault(&f));
        CHECK(f.code == EXCEPTION_INT_DIVIDE_BY_ZERO);
        REQUIRE_OK(kcDeviceReset());
        REQUIRE(llm::registerKernels() == Error::Success);
    }
}
