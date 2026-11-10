// End-to-end: allocate -> copy -> launch Booth kernel -> copy back -> compare.

#include "harness.hpp"
#include "test_kernels.hpp"

#include <kernelcore/kernels.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using kc::CopyKind;
using kc::Error;

extern "C" {
float kc_crt_sinf(float);
float kc_crt_cosf(float);
float kc_crt_exp2f(float);
float kc_crt_log2f(float);
}

namespace {

template <class T>
struct DeviceBuffer {
    T* p = nullptr;
    explicit DeviceBuffer(size_t n) {
        if (kcMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)) != Error::Success) p = nullptr;
    }
    ~DeviceBuffer() {
        if (p) kcFree(p);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

float valueA(size_t i) { return static_cast<float>(i % 1000) * 0.25f - 17.0f; }
float valueB(size_t i) { return static_cast<float>((i * 7919) % 4093) * 0.5f; }

bool runVectorAdd(int n) {
    const size_t N = static_cast<size_t>(n);
    std::vector<float> a(N), b(N), out(N, -1.0f);
    for (size_t i = 0; i < N; ++i) {
        a[i] = valueA(i);
        b[i] = valueB(i);
    }
    DeviceBuffer<float> da(N), db(N), dout(N);
    if (!da.p || !db.p || !dout.p) return false;
    if (kcMemcpy(da.p, a.data(), N * 4, CopyKind::HostToDevice) != Error::Success) return false;
    if (kcMemcpy(db.p, b.data(), N * 4, CopyKind::HostToDevice) != Error::Success) return false;
    const Error e = kc::kernels::launchVectorAdd(da.p, db.p, dout.p, n);
    if (e != Error::Success) {
        th::fail(__FILE__, __LINE__, "launch n=" + std::to_string(n) + ": " + th::describe(e));
        return false;
    }
    if (kcMemcpy(out.data(), dout.p, N * 4, CopyKind::DeviceToHost) != Error::Success) return false;
    for (size_t i = 0; i < N; ++i) {
        const float want = a[i] + b[i];
        if (std::bit_cast<uint32_t>(out[i]) != std::bit_cast<uint32_t>(want)) {
            th::fail(__FILE__, __LINE__, "n=" + std::to_string(n) + " mismatch at " + std::to_string(i));
            return false;
        }
    }
    return true;
}

int ulpDistance(float x, float y) {
    if (std::isnan(x) || std::isnan(y)) return (std::isnan(x) && std::isnan(y)) ? 0 : 1 << 30;
    auto key = [](float f) {
        const int32_t i = std::bit_cast<int32_t>(f);
        return i < 0 ? INT32_MIN - i : i;
    };
    const int64_t d = static_cast<int64_t>(key(x)) - key(y);
    return static_cast<int>(d < 0 ? -d : d);
}

} // namespace

TEST_CASE(vector_add_boundary_sizes_bit_exact) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    for (int n : {1, 7, 63, 64, 65, 256, 257, 4096, 100003, 1 << 20, 1 << 24})
        CHECK(runVectorAdd(n));
}

TEST_CASE(vector_add_zero_count_is_noop_and_negative_is_rejected) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    DeviceBuffer<float> d(4);
    REQUIRE(d.p);
    CHECK_OK(kc::kernels::launchVectorAdd(d.p, d.p, d.p, 0));
    CHECK_ERR(kc::kernels::launchVectorAdd(d.p, d.p, d.p, -1), Error::InvalidValue);
}

TEST_CASE(vector_add_rejects_ranges_beyond_allocations) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    DeviceBuffer<float> a(100), b(100), out(99);
    REQUIRE(a.p && b.p && out.p);
    CHECK_ERR(kc::kernels::launchVectorAdd(a.p, b.p, out.p, 100), Error::InvalidPointer);
    CHECK(std::strstr(kcGetLastErrorDetail(), "'out'") != nullptr);
    std::vector<float> host(100);
    CHECK_ERR(kc::kernels::launchVectorAdd(host.data(), b.p, a.p, 100), Error::InvalidPointer);
    // interior sub-range is fine
    CHECK_OK(kc::kernels::launchVectorAdd(a.p + 10, b.p + 10, out.p + 10, 89));
    // nothing executed after rejection: context is still healthy
    void* p = nullptr;
    CHECK_OK(kcMalloc(&p, 4));
    CHECK_OK(kcFree(p));
}

TEST_CASE(float_scalar_arguments_use_xmm_registers) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1000;
    std::vector<float> x(n), out(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(i) - 500.0f;
    DeviceBuffer<float> dx(n), dout(n);
    REQUIRE(dx.p && dout.p);
    CHECK_OK(kcMemcpy(dx.p, x.data(), n * 4, CopyKind::HostToDevice));
    const testkernels::ScaleAddArgs args{dx.p, dout.p, 2.5f, -0.75f, n};
    CHECK_OK(kcLaunchKernel("scale_add", &args));
    CHECK_OK(kcMemcpy(out.data(), dout.p, n * 4, CopyKind::DeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const volatile float prod = x[i] * 2.5f;  // same two roundings as the kernel (no FMA)
        ok = ok && out[i] == prod + -0.75f;
    }
    CHECK(ok);
}

TEST_CASE(libm_calls_go_through_sysv_shims) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 4096;
    std::vector<float> x(n), out(n);
    for (int i = 0; i < n; ++i) x[i] = -1.0f + 2.0f * static_cast<float>(i) / n;
    DeviceBuffer<float> dx(n), dout(n);
    REQUIRE(dx.p && dout.p);
    CHECK_OK(kcMemcpy(dx.p, x.data(), n * 4, CopyKind::HostToDevice));
    const testkernels::TrigMixArgs args{dx.p, dout.p, n};
    CHECK_OK(kcLaunchKernel("trig_mix", &args));
    CHECK_OK(kcMemcpy(out.data(), dout.p, n * 4, CopyKind::DeviceToHost));
    // Reference 1: the exact out-of-line CRT wrappers the shims call, with Booth 0.6.1's
    // argument lowering for sin/cos: sinf((x * 1/(2*pi)) * 2*pi) in float (constants
    // 0x3e22f983, 0x40c90fdb in the object). Must match bit for bit: this proves arguments
    // and results cross the System V/Win64 boundary unchanged.
    // Reference 2: double-precision math; only a relative tolerance applies.
    const float inv_2pi = std::bit_cast<float>(0x3e22f983u);
    const float two_pi = std::bit_cast<float>(0x40c90fdbu);
    int worst_exact = 0;
    double worst_rel = 0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        const volatile float turns = v * inv_2pi;  // two separate float roundings, as Booth does
        const volatile float arg = turns * two_pi;
        const float ref = ((kc_crt_sinf(arg) + kc_crt_cosf(arg)) + kc_crt_exp2f(v)) + kc_crt_log2f(v + 2.0f);
        worst_exact = std::max(worst_exact, ulpDistance(out[i], ref));
        const double precise = std::sin(double{v}) + std::cos(double{v}) + std::exp2(double{v}) + std::log2(double{v} + 2.0);
        worst_rel = std::max(worst_rel, std::abs((out[i] - precise) / precise));
    }
    std::printf("  ULP distance vs shim targets: %d, max relative error vs double reference: %.2e\n", worst_exact,
                worst_rel);
    CHECK(worst_exact == 0);
    CHECK(worst_rel < 1e-5);
}

TEST_CASE(int_kernel_with_negative_values) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 257;
    std::vector<int> a(n), out(n);
    for (int i = 0; i < n; ++i) a[i] = (i - 128) * 1000;
    DeviceBuffer<int> da(n), dout(n);
    REQUIRE(da.p && dout.p);
    CHECK_OK(kcMemcpy(da.p, a.data(), n * 4, CopyKind::HostToDevice));
    const testkernels::IntDivArgs args{da.p, dout.p, -7, n};  // negative int argument: sign extension
    CHECK_OK(kcLaunchKernel("int_div", &args));
    CHECK_OK(kcMemcpy(out.data(), dout.p, n * 4, CopyKind::DeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && out[i] == a[i] / -7;
    CHECK(ok);
}

TEST_CASE(recursive_device_function_within_stack) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    DeviceBuffer<int> d(4);
    REQUIRE(d.p);
    const testkernels::DeepRecursionArgs args{d.p, 1000, 4};
    CHECK_OK(kcLaunchKernel("deep_recursion", &args));
    int out[4] = {};
    CHECK_OK(kcMemcpy(out, d.p, sizeof out, CopyKind::DeviceToHost));
    CHECK(out[0] == 1000 && out[3] == 1000);
}
