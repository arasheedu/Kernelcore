// LLM decode kernels (kernels/llm/*.cu) under Kernelcore: each against the plain C++ reference,
// chunked == serial, argument validation, and one full decode token of a tiny model.

#include "harness.hpp"

#include "llm_kernels.hpp"
#include "llm_model.hpp"
#include "llm_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

using kc::CopyKind;
using kc::Error;

namespace {

struct LlmFixture {
    RuntimeFixture rt;
    bool ok = false;
    LlmFixture() { ok = rt.ok && llm::registerKernels() == Error::Success; }
};

template <class T>
T* devCopy(const std::vector<T>& h) {
    void* p = nullptr;
    if (kcMalloc(&p, h.size() * sizeof(T)) != Error::Success) return nullptr;
    if (kcMemcpy(p, h.data(), h.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}

template <class T>
std::vector<T> hostCopy(const T* d, size_t n) {
    std::vector<T> h(n);
    if (kcMemcpy(h.data(), d, n * sizeof(T), CopyKind::DeviceToHost) != Error::Success) h.clear();
    return h;
}

std::vector<float> randomFloats(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& f : v) f = d(rng);
    return v;
}

// Largest |a - b| relative to the largest |b|: robust where individual values cross zero.
double relDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double m = 0, scale = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) return 1e30;
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
        scale = std::max(scale, std::abs(double(b[i])));
    }
    return scale > 0 ? m / scale : m;
}

bool bitEqual(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}

unsigned workers() { return std::max(2u, std::thread::hardware_concurrency()); }

} // namespace

TEST_CASE(matvec_q8_matches_reference_and_chunks) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (auto [rows, k] : {std::pair{1, 32}, std::pair{97, 64}, std::pair{1000, 256}, std::pair{4099, 96}}) {
        std::mt19937 rng(unsigned(rows * 31 + k));
        std::vector<int8_t> q(size_t(rows) * k);
        for (auto& v : q) v = static_cast<int8_t>(int(rng() % 255) - 127);
        const auto s = randomFloats(q.size() / 32, 7, 0.001f, 0.01f);
        const auto x = randomFloats(size_t(k), 8);
        auto* dq = devCopy(q);
        auto* ds = devCopy(s);
        auto* dx = devCopy(x);
        auto* dy = devCopy(std::vector<float>(size_t(rows), 0.0f));
        auto* dd = devCopy(std::vector<int>{rows, k});
        REQUIRE(dq && ds && dx && dy && dd);
        std::vector<float> ref(static_cast<size_t>(rows));
        llm::ref::matvecQ8(q.data(), s.data(), x.data(), ref.data(), rows, k);

        const llm::MatvecQ8Args a{dq, ds, dx, dy, dd, rows, k};
        CHECK_OK(kcLaunchKernel("matvec_q8", &a));
        const auto serial = hostCopy(dy, size_t(rows));
        CHECK(bitEqual(serial, ref));  // same loop order, no FMA: bit-exact
        CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(kcLaunchKernelChunked("matvec_q8", &a, workers(), 64));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), serial));
        for (void* p : {static_cast<void*>(dq), static_cast<void*>(ds), static_cast<void*>(dx),
                        static_cast<void*>(dy), static_cast<void*>(dd)})
            CHECK_OK(kcFree(p));
    }
}

// The 8-row interleaved kernel matches its C++ reference exactly, and matvec_q4 on the same
// (row-major) weights up to summation order.
TEST_CASE(matvec_q4x8_matches_reference_and_q4) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (auto [rows, k] : {std::pair{8, 32}, std::pair{96, 64}, std::pair{1000, 256}}) {
        std::mt19937 rng(unsigned(rows + k));
        std::vector<uint8_t> q(size_t(rows) * k / 2), q8(q.size());
        for (auto& v : q) v = static_cast<uint8_t>(rng());
        const auto s = randomFloats(size_t(rows) * k / 32, 21, 0.001f, 0.01f);
        std::vector<float> s8(s.size());
        llm::ref::repackQ4x8(q.data(), s.data(), q8.data(), s8.data(), rows, k);
        const auto x = randomFloats(size_t(k), 22);
        std::vector<float> want(static_cast<size_t>(rows)), want8(static_cast<size_t>(rows));
        llm::ref::matvecQ4(q.data(), s.data(), x.data(), want.data(), rows, k);
        llm::ref::matvecQ4x8(q8.data(), s8.data(), x.data(), want8.data(), rows, k);
        CHECK(relDiff(want8, want) < 1e-5);  // 4 partial sums vs one running sum
        auto* dq = devCopy(q8);
        auto* ds = devCopy(s8);
        auto* dx = devCopy(x);
        auto* dy = devCopy(std::vector<float>(size_t(rows), 0.0f));
        auto* dd = devCopy(std::vector<int>{rows, k});
        REQUIRE(dq && ds && dx && dy && dd);
        const llm::MatvecQ4Args a{dq, ds, dx, dy, dd, rows, k};
        CHECK_OK(kcLaunchKernel("matvec_q4x8", &a));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want8));
        CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(kcLaunchKernelChunked("matvec_q4x8", &a, workers(), 8));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want8));
    }
    // rows must be a multiple of 8
    auto* buf = devCopy(std::vector<float>(1024, 0.0f));
    auto* q = devCopy(std::vector<uint8_t>(4096, 0));
    auto* dims = devCopy(std::vector<int>{12, 32});
    REQUIRE(buf && q && dims);
    const llm::MatvecQ4Args bad{q, buf, buf, buf, dims, 12, 32};
    CHECK_ERR(kcLaunchKernel("matvec_q4x8", &bad), Error::InvalidValue);
}

// 8-bit activations + integer dot products: exact against the C++ reference, and within
// quantization error of the float kernel on the same weights.
TEST_CASE(matvec_q4x8_q8_matches_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (auto [rows, k] : {std::pair{8, 32}, std::pair{96, 64}, std::pair{1000, 512}, std::pair{1008, 256}}) {
        std::mt19937 rng(unsigned(rows * 7 + k));
        std::vector<uint8_t> q(size_t(rows) * k / 2);
        for (auto& v : q) v = static_cast<uint8_t>(rng());
        const auto s = randomFloats(size_t(rows) * k / 32, 31, 0.001f, 0.01f);
        std::vector<int> qw(size_t(rows) * k * 5 / 32);
        llm::ref::repackQ4x8Q8(q.data(), s.data(), qw.data(), rows, k);
        auto x = randomFloats(size_t(k), 32, -3, 3);
        x[0] = 0.0f;  // a zero, and an all-zero block when k allows it
        if (k >= 64) std::fill(x.begin() + 32, x.begin() + 64, 0.0f);
        const int nb = k / 32;
        std::vector<int> xq(size_t(nb) * 8);
        std::vector<float> xd(size_t(nb) * 2), want(static_cast<size_t>(rows)), flt(static_cast<size_t>(rows));
        llm::ref::quantizeQ8x4(x.data(), xq.data(), xd.data(), nb);
        llm::ref::matvecQ4x8Q8(qw.data(), xq.data(), xd.data(), want.data(), rows, k);
        llm::ref::matvecQ4(q.data(), s.data(), x.data(), flt.data(), rows, k);
        CHECK(relDiff(want, flt) < 3e-2);  // 8-bit activation rounding

        auto* dx = devCopy(x);
        auto* dqw = devCopy(qw);
        auto* dxq = devCopy(std::vector<int>(xq.size(), 0));
        auto* dxd = devCopy(std::vector<float>(xd.size(), 0.0f));
        auto* dy = devCopy(std::vector<float>(size_t(rows), 0.0f));
        auto* dd = devCopy(std::vector<int>{rows, k});
        REQUIRE(dx && dqw && dxq && dxd && dy && dd);
        const llm::QuantizeArgs qa{dx, dxq, dxd, nb};
        CHECK_OK(kcLaunchKernel("quantize_q8x4", &qa));
        CHECK(hostCopy(dxq, xq.size()) == xq);
        CHECK(bitEqual(hostCopy(dxd, xd.size()), xd));
        const llm::MatvecQ4Q8Args a{dqw, dxq, dxd, dy, dd, rows, k};
        CHECK_OK(kcLaunchKernel("matvec_q4x8_q8", &a));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
        CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(kcLaunchKernelChunked("matvec_q4x8_q8", &a, workers(), 8));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
        if (rows % 16 == 0) {  // two rows per thread: the same bits
            CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
            CHECK_OK(kcLaunchKernel("matvec_q4x8_q8x2", &a));
            CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
            CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
            CHECK_OK(kcLaunchKernelChunked("matvec_q4x8_q8x2", &a, workers(), 8));
            CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
        } else {
            CHECK_ERR(kcLaunchKernel("matvec_q4x8_q8x2", &a), Error::InvalidValue);
        }
    }
}


// GGUF K-quant matvecs: random Q4_K / Q6_K blocks, repacked; bit-exact against the C++ mirror,
// close to the float dot product of ggml's own dequantization.
TEST_CASE(kquant_matvecs_match_reference_and_dequantized_floats) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (int q6 = 0; q6 < 2; ++q6) {
        for (auto [rows, k] : {std::pair{8, 256}, std::pair{40, 768}}) {
            const int nsb = k / 256, bb = q6 ? 210 : 144, words = q6 ? 53 : 37;
            std::mt19937 rng(unsigned(rows * 31 + k + q6));
            std::vector<uint8_t> src(size_t(rows) * nsb * bb);
            for (auto& v : src) v = static_cast<uint8_t>(rng());
            auto setHalf = [&](uint8_t* p) {  // a positive fp16 scale in [2^-9, 2^-6)
                const uint16_t h = static_cast<uint16_t>(0x1800 + rng() % 0x0C00);
                p[0] = uint8_t(h & 0xFF), p[1] = uint8_t(h >> 8);
            };
            for (int r = 0; r < rows * nsb; ++r) {
                uint8_t* b = src.data() + size_t(r) * bb;
                if (q6) setHalf(b + 208);
                else setHalf(b), setHalf(b + 2);
            }
            std::vector<int> qw(size_t(rows) * nsb * words);
            if (q6) llm::ref::repackQ6K(src.data(), rows, k, qw.data());
            else llm::ref::repackQ4K(src.data(), rows, k, qw.data());

            const auto x = randomFloats(size_t(k), 61, -3, 3);
            const int nb = k / 32;
            std::vector<int> xq(size_t(nb) * 8);
            std::vector<float> xd(static_cast<size_t>(nb) * 2), want(static_cast<size_t>(rows)), flt(static_cast<size_t>(rows)),
                row(static_cast<size_t>(k));
            llm::ref::quantizeQ8x4(x.data(), xq.data(), xd.data(), nb);
            if (q6) llm::ref::matvecQ6KQ8(qw.data(), xq.data(), xd.data(), want.data(), rows, k);
            else llm::ref::matvecQ4KQ8(qw.data(), xq.data(), xd.data(), want.data(), rows, k);
            for (int r = 0; r < rows; ++r) {
                const uint8_t* rs = src.data() + size_t(r) * nsb * bb;
                if (q6) llm::ref::dequantRowQ6K(rs, k, row.data());
                else llm::ref::dequantRowQ4K(rs, k, row.data());
                double s = 0;
                for (int i = 0; i < k; ++i) s += double(row[size_t(i)]) * x[size_t(i)];
                flt[size_t(r)] = float(s);
            }
            if (!(relDiff(want, flt) < 3e-2))
                th::fail(__FILE__, __LINE__, std::string(q6 ? "q6_K" : "q4_K") + " vs dequantized floats: " +
                                                 std::to_string(relDiff(want, flt)));

            auto* dqw = devCopy(qw);
            auto* dxq = devCopy(xq);
            auto* dxd = devCopy(xd);
            auto* dy = devCopy(std::vector<float>(size_t(rows), 0.0f));
            auto* dd = devCopy(std::vector<int>{rows, k});
            REQUIRE(dqw && dxq && dxd && dy && dd);
            const char* kn = q6 ? "matvec_q6k_q8" : "matvec_q4k_q8";
            const llm::MatvecQ4Q8Args a{dqw, dxq, dxd, dy, dd, rows, k};
            CHECK_OK(kcLaunchKernel(kn, &a));
            CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
            CHECK_OK(kcMemset(dy, 0xFF, size_t(rows) * 4));
            CHECK_OK(kcLaunchKernelChunked(kn, &a, workers(), 8));
            CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
        }
    }
    // rope with frequency factors
    const int pairs = 64, heads = 3;
    const auto x = randomFloats(size_t(heads) * pairs * 2, 62, -1, 1);
    const auto ff = randomFloats(size_t(pairs), 63, 1.0f, 8.0f);
    std::vector<float> rx = x;
    llm::ref::ropeFF(rx.data(), ff.data(), pairs, 77, heads * pairs, std::log2(500000.0f));
    auto* dx = devCopy(x);
    auto* dff = devCopy(ff);
    REQUIRE(dx && dff);
    const llm::RopeFFArgs ra{dx, dff, pairs, 77, heads * pairs, std::log2(500000.0f)};
    CHECK_OK(kcLaunchKernel("rope_ff", &ra));
    CHECK(bitEqual(hostCopy(dx, x.size()), rx));
}


// Multi-column K-quant matmuls (prompt processing): quantize_cols fills one scratch slot per
// column exactly as quantize_q8x4; matmul_q4k_q8x4 / matmul_q6k_q8x4 give, per column, the
// mirror matvec's bits, with 4- or 8-column tiles. Column counts cover full and short tiles; chunked launches too.
TEST_CASE(kquant_matmuls_equal_per_column_matvecs) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int rows = 40, k = 768, nsb = k / 256, nb = k / 32, slot = llm::scratchInts(nb);
    for (int q6 = 0; q6 < 2; ++q6) {
        const int bb = q6 ? 210 : 144, words = q6 ? 53 : 37;
        std::mt19937 rng(unsigned(901 + q6));
        std::vector<uint8_t> src(size_t(rows) * nsb * bb);
        for (auto& v : src) v = static_cast<uint8_t>(rng());
        auto setHalf = [&](uint8_t* p) {
            const uint16_t h = static_cast<uint16_t>(0x1800 + rng() % 0x0C00);
            p[0] = uint8_t(h & 0xFF), p[1] = uint8_t(h >> 8);
        };
        for (int r = 0; r < rows * nsb; ++r) {
            uint8_t* b = src.data() + size_t(r) * bb;
            if (q6) setHalf(b + 208);
            else setHalf(b), setHalf(b + 2);
        }
        std::vector<int> qw(size_t(rows) * nsb * words);
        if (q6) llm::ref::repackQ6K(src.data(), rows, k, qw.data());
        else llm::ref::repackQ4K(src.data(), rows, k, qw.data());
        auto* dqw = devCopy(qw);
        REQUIRE(dqw);
        for (int ncols : {1, 3, 4, 7, 8, 9, 13}) {
            const auto x = randomFloats(size_t(k) * ncols, unsigned(950 + ncols), -3, 3);
            std::vector<int> scratch(size_t(slot) * ncols, 0);
            std::vector<float> want(size_t(rows) * ncols);
            for (int c = 0; c < ncols; ++c) {
                int* s = scratch.data() + size_t(c) * slot;
                llm::ref::quantizeQ8x4(x.data() + size_t(c) * k, s, llm::scratchScales(s, nb), nb);
                if (q6) llm::ref::matvecQ6KQ8(qw.data(), s, llm::scratchScales(s, nb), want.data() + size_t(c) * rows, rows, k);
                else llm::ref::matvecQ4KQ8(qw.data(), s, llm::scratchScales(s, nb), want.data() + size_t(c) * rows, rows, k);
            }
            auto* dx = devCopy(x);
            auto* dq = devCopy(std::vector<int>(scratch.size(), 0));
            auto* dy = devCopy(std::vector<float>(want.size(), 0.0f));
            auto* dqd = devCopy(std::vector<int>{nb, ncols});
            REQUIRE(dx && dq && dy && dqd);
            const llm::QuantizeColsArgs qa{dx, dq, dqd, k, ncols};
            CHECK_OK(kcLaunchKernelChunked("quantize_cols", &qa, workers(), 8));
            const auto got = hostCopy(dq, scratch.size());
            bool same = true;  // words and {scale, sum}; the partial-sum floats are not written
            for (int c = 0; c < ncols; ++c)
                for (int i = 0; i < nb * 10; ++i) same = same && got[size_t(c) * slot + i] == scratch[size_t(c) * slot + i];
            CHECK(same);
            for (const char* kn : {q6 ? "matmul_q6k_q8x4" : "matmul_q4k_q8x4", q6 ? "matmul_q6k_q8x8" : "matmul_q4k_q8x8"})
            for (int rb : {1, 2, 3, 8, 1000}) {  // 5 row groups: partial last blocks, tile-major
                auto* dmd = devCopy(std::vector<int>{rows, k, ncols, rb});
                REQUIRE(dmd);
                const llm::MatmulKQ8Args a{dqw, dq, dy, dmd, rows, k, ncols, rb};
                CHECK_OK(kcMemset(dy, 0xFF, want.size() * 4));
                CHECK_OK(kcLaunchKernel(kn, &a));
                CHECK(bitEqual(hostCopy(dy, want.size()), want));
                CHECK_OK(kcMemset(dy, 0xFF, want.size() * 4));
                CHECK_OK(kcLaunchKernelChunked(kn, &a, workers(), 8));
                CHECK(bitEqual(hostCopy(dy, want.size()), want));
                if (ncols > 1) {  // dims disagree with the arguments
                    const llm::MatmulKQ8Args bad{dqw, dq, dy, dmd, rows, k, ncols - 1, rb};
                    CHECK_ERR(kcLaunchKernel(kn, &bad), Error::InvalidValue);
                }
            }
        }
    }
}

// Activations quantized per 256 values (quantize_cols_k) and the matmul_*_q8kx* kernels: bit-identical
// to the C++ reference, and as close to the exact float product as the per-32 path.
TEST_CASE(kquant_q8k_matmuls_match_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int rows = 40, k = 768, nsb = k / 256, nb = k / 32, slot = llm::scratchKInts(nsb);
    for (int q6 = 0; q6 < 2; ++q6) {
        const int bb = q6 ? 210 : 144, words = q6 ? 53 : 37;
        std::mt19937 rng(unsigned(1901 + q6));
        std::vector<uint8_t> src(size_t(rows) * nsb * bb);
        for (auto& v : src) v = static_cast<uint8_t>(rng());
        auto setHalf = [&](uint8_t* p) {
            const uint16_t h = static_cast<uint16_t>(0x1800 + rng() % 0x0C00);
            p[0] = uint8_t(h & 0xFF), p[1] = uint8_t(h >> 8);
        };
        for (int r = 0; r < rows * nsb; ++r) {
            uint8_t* b = src.data() + size_t(r) * bb;
            if (q6) setHalf(b + 208);
            else setHalf(b), setHalf(b + 2);
        }
        std::vector<int> qw(size_t(rows) * nsb * words);
        if (q6) llm::ref::repackQ6K(src.data(), rows, k, qw.data());
        else llm::ref::repackQ4K(src.data(), rows, k, qw.data());
        std::vector<float> wf(size_t(rows) * k);  // dequantized weights, for the exact product
        for (int r = 0; r < rows; ++r) {
            if (q6) llm::ref::dequantRowQ6K(src.data() + size_t(r) * nsb * bb, k, wf.data() + size_t(r) * k);
            else llm::ref::dequantRowQ4K(src.data() + size_t(r) * nsb * bb, k, wf.data() + size_t(r) * k);
        }
        auto* dqw = devCopy(qw);
        REQUIRE(dqw);
        for (int ncols : {1, 3, 4, 7, 8, 9, 13}) {
            const auto x = randomFloats(size_t(k) * ncols, unsigned(1950 + ncols), -3, 3);
            std::vector<int>   want_q(size_t(slot) * ncols);
            std::vector<float> want(size_t(rows) * ncols);
            double err_k = 0, err_32 = 0, scale = 0;
            for (int c = 0; c < ncols; ++c) {
                int* s = want_q.data() + size_t(c) * slot;
                llm::ref::quantizeQ8K(x.data() + size_t(c) * k, s, nsb);
                float* yc = want.data() + size_t(c) * rows;
                if (q6) llm::ref::matvecQ6KQ8K(qw.data(), s, yc, rows, k);
                else llm::ref::matvecQ4KQ8K(qw.data(), s, yc, rows, k);
                std::vector<int> s32(size_t(llm::scratchInts(nb)));  // the per-32 path, for comparison
                std::vector<float> y32(rows);
                llm::ref::quantizeQ8x4(x.data() + size_t(c) * k, s32.data(), llm::scratchScales(s32.data(), nb), nb);
                if (q6) llm::ref::matvecQ6KQ8(qw.data(), s32.data(), llm::scratchScales(s32.data(), nb), y32.data(), rows, k);
                else llm::ref::matvecQ4KQ8(qw.data(), s32.data(), llm::scratchScales(s32.data(), nb), y32.data(), rows, k);
                for (int r = 0; r < rows; ++r) {
                    double exact = 0;
                    for (int i = 0; i < k; ++i) exact += double(wf[size_t(r) * k + i]) * x[size_t(c) * k + i];
                    scale = std::max(scale, std::fabs(exact));
                    err_k = std::max(err_k, std::fabs(yc[r] - exact));
                    err_32 = std::max(err_32, std::fabs(y32[r] - exact));
                }
            }
            CHECK(err_k < 0.02 * scale);       // 8-bit activations: well under 2% of the largest output
            CHECK(err_k < 4 * err_32 + 1e-6);  // coarser scales: same order of error as per 32
            auto* dx = devCopy(x);
            auto* dq = devCopy(std::vector<int>(want_q.size(), 0));
            auto* dy = devCopy(std::vector<float>(want.size(), 0.0f));
            auto* dqd = devCopy(std::vector<int>{nsb, ncols});
            REQUIRE(dx && dq && dy && dqd);
            const llm::QuantizeColsArgs qa{dx, dq, dqd, k, ncols};
            CHECK_OK(kcLaunchKernelChunked("quantize_cols_k", &qa, workers(), 8));
            CHECK(hostCopy(dq, want_q.size()) == want_q);
            for (const char* kn : {q6 ? "matmul_q6k_q8kx4" : "matmul_q4k_q8kx4", q6 ? "matmul_q6k_q8kx8" : "matmul_q4k_q8kx8"})
                for (int rb : {1, 3, 1000}) {
                    auto* dmd = devCopy(std::vector<int>{rows, k, ncols, rb});
                    REQUIRE(dmd);
                    const llm::MatmulKQ8Args a{dqw, dq, dy, dmd, rows, k, ncols, rb};
                    CHECK_OK(kcMemset(dy, 0xFF, want.size() * 4));
                    CHECK_OK(kcLaunchKernel(kn, &a));
                    CHECK(bitEqual(hostCopy(dy, want.size()), want));
                    CHECK_OK(kcMemset(dy, 0xFF, want.size() * 4));
                    CHECK_OK(kcLaunchKernelChunked(kn, &a, workers(), 8));
                    CHECK(bitEqual(hostCopy(dy, want.size()), want));
                }
        }
    }
}

// ggml ops for the llama.cpp backend: broadcast multiply and rope with positions from a tensor.
TEST_CASE(ggml_op_kernels_match_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int n = 3072, rows = 3;
    const auto x = randomFloats(size_t(n) * rows, 71, -2, 2), w = randomFloats(size_t(n), 72, 0.5f, 1.5f);
    std::vector<float> ry(x.size());
    llm::ref::mulRows(x.data(), w.data(), ry.data(), n, n * rows);
    auto *dx = devCopy(x), *dw = devCopy(w), *dy = devCopy(std::vector<float>(x.size(), 0.0f));
    REQUIRE(dx && dw && dy);
    const llm::MulRowsArgs ma{dx, dw, dy, n, n * rows};
    CHECK_OK(kcLaunchKernel("mul_rows", &ma));
    CHECK(bitEqual(hostCopy(dy, ry.size()), ry));
    const llm::MulRowsArgs bad{dx, dw, dy, 100, 300};  // n must be a multiple of 8
    CHECK_ERR(kcLaunchKernel("mul_rows", &bad), Error::InvalidValue);

    const int hd = 128, nh = 6, nt = 3, pairs = hd / 2;
    const auto q = randomFloats(size_t(hd) * nh * nt, 73, -1, 1), ff = randomFloats(size_t(pairs), 74, 1.0f, 8.0f);
    const std::vector<int> pos{5, 300, 4097};
    for (bool neox : {false, true}) {  // NORM (Llama) and NEOX (Qwen) pairs
        std::vector<float> rq(q.size());
        llm::ref::ropeGgml(q.data(), rq.data(), pos.data(), ff.data(), hd, nh, nt, std::log2(500000.0f), neox);
        auto *dq = devCopy(q), *dout = devCopy(std::vector<float>(q.size(), 0.0f)), *dff = devCopy(ff);
        auto *dpos = devCopy(pos), *ddims = devCopy(std::vector<int>{pairs, nh, pairs * nt, neox ? 1 : 2, neox ? pairs : 1});
        REQUIRE(dq && dout && dff && dpos && ddims);
        const llm::RopeGgmlArgs ra{dq, dout, dpos, dff, ddims, hd, nh, nt, std::log2(500000.0f), neox};
        CHECK_OK(kcLaunchKernel("rope_ggml", &ra));
        CHECK(bitEqual(hostCopy(dout, rq.size()), rq));
        const llm::RopeGgmlArgs wrong{dq, dout, dpos, dff, ddims, hd, nh, nt, std::log2(500000.0f), !neox};
        CHECK_ERR(kcLaunchKernel("rope_ggml", &wrong), Error::InvalidValue);  // dims disagree with the mode
    }
}


// ggml fusions: residual add + RMS norm + weight + quantize, and SwiGLU + quantize; chunked or not.
TEST_CASE(ggml_fused_kernels_match_reference_and_chunk) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int n = 3072, rows = 2, slot = llm::scratchInts(n / 32);
    const auto a = randomFloats(size_t(n) * rows, 81, -2, 2), b = randomFloats(size_t(n) * rows, 82, -2, 2);
    const auto w = randomFloats(size_t(n), 83, 0.9f, 1.1f);
    std::vector<float> rout(a.size()), ry(a.size());
    std::vector<int> rq(size_t(slot) * rows, 0);
    llm::ref::addSumsqF(a.data(), b.data(), rout.data(), rq.data(), n, rows);
    llm::ref::normMulQ8(rout.data(), w.data(), ry.data(), rq.data(), n, rows, 1e-5f);
    std::vector<float> ry2(a.size());
    std::vector<int> rq2(size_t(slot) * rows, 0);
    llm::ref::sumsqPart(a.data(), rq2.data(), n, rows);
    llm::ref::normMulQ8(a.data(), w.data(), ry2.data(), rq2.data(), n, rows, 1e-5f);
    for (bool chunked : {false, true}) {
        auto *da = devCopy(a), *db = devCopy(b), *dw = devCopy(w);
        auto *dout = devCopy(std::vector<float>(a.size())), *dy = devCopy(std::vector<float>(a.size()));
        auto* dq = devCopy(std::vector<int>(rq.size()));
        auto* dd = devCopy(std::vector<int>{n / 32, n, rows});
        REQUIRE(da && db && dw && dout && dy && dq && dd);
        auto run = [&](const char* k, const auto& args) {
            return chunked ? kcLaunchKernelChunked(k, &args, workers(), 16) : kcLaunchKernel(k, &args);
        };
        CHECK_OK(run("add_sumsq_f", llm::AddSumsqFArgs{da, db, dout, dq, dd, n, rows}));
        CHECK_OK(run("norm_mul_q8_f", llm::NormMulQ8Args{dout, dw, dy, dq, dd, n, rows, 1e-5f}));
        CHECK(bitEqual(hostCopy(dout, rout.size()), rout));
        CHECK(bitEqual(hostCopy(dy, ry.size()), ry));
        CHECK(hostCopy(dq, rq.size()) == rq);
        CHECK_OK(run("sumsq_part", llm::SumsqPartArgs{da, dq, dd, n, rows}));
        CHECK_OK(run("norm_mul_q8_f", llm::NormMulQ8Args{da, dw, dy, dq, dd, n, rows, 1e-5f}));
        CHECK(bitEqual(hostCopy(dy, ry2.size()), ry2));
        CHECK(hostCopy(dq, rq2.size()) == rq2);
    }
    // SwiGLU + quantize equals silu_mul then quantize_q8x4
    const int m = 8192, cols = 2, mslot = llm::scratchInts(m / 32);
    const auto g = randomFloats(size_t(m) * cols, 84, -6, 6), u = randomFloats(size_t(m) * cols, 85, -2, 2);
    std::vector<float> rs(g.size());
    std::vector<int> rsq(size_t(mslot) * cols, 0);
    llm::ref::siluMulQ8F(g.data(), u.data(), rs.data(), rsq.data(), m, cols);
    for (bool chunked : {false, true}) {
        auto *dg = devCopy(g), *du = devCopy(u), *dy = devCopy(std::vector<float>(g.size()));
        auto* dq = devCopy(std::vector<int>(rsq.size()));
        auto* dd = devCopy(std::vector<int>{m, cols});
        REQUIRE(dg && du && dy && dq && dd);
        const llm::SiluMulQ8FArgs sa{dg, du, dy, dq, dd, m, cols};
        CHECK_OK(chunked ? kcLaunchKernelChunked("silu_mul_q8x4f", &sa, workers(), 16) : kcLaunchKernel("silu_mul_q8x4f", &sa));
        CHECK(bitEqual(hostCopy(dy, rs.size()), rs));
        CHECK(hostCopy(dq, rsq.size()) == rsq);
    }
    // Per 256 values (batches): norm_mul_q8k_f and silu_mul_q8kf; 5 rows so that chunks split rows.
    {
        const int kr = 5, off = kr * llm::scratchKInts(n / 256);
        const auto ka = randomFloats(size_t(n) * kr, 86, -2, 2), kb = randomFloats(size_t(n) * kr, 87, -2, 2);
        std::vector<float> kout(ka.size()), ky(ka.size());
        std::vector<int> ksq(size_t(slot) * kr, 0), kq(size_t(off), 0);
        llm::ref::addSumsqF(ka.data(), kb.data(), kout.data(), ksq.data(), n, kr);
        llm::ref::normMulQ8K(kout.data(), w.data(), ky.data(), kq.data(), ksq.data(), n, kr, 1e-5f);
        const auto ks = randomFloats(size_t(m) * kr, 88, -6, 6), ku = randomFloats(size_t(m) * kr, 89, -2, 2);
        std::vector<float> ksy(ks.size());
        std::vector<int> ksq8(size_t(kr) * llm::scratchKInts(m / 256), 0);
        llm::ref::siluMulQ8K(ks.data(), ku.data(), ksy.data(), ksq8.data(), m, kr);
        for (bool chunked : {false, true}) {
            auto *da = devCopy(ka), *db = devCopy(kb), *dw = devCopy(w);
            auto *dout = devCopy(std::vector<float>(ka.size())), *dy = devCopy(std::vector<float>(ka.size()));
            auto* dq = devCopy(std::vector<int>(size_t(off) + ksq.size()));
            auto* dd = devCopy(std::vector<int>{n / 32, n, kr});
            auto* ddk = devCopy(std::vector<int>{n / 256, n, kr, off});
            REQUIRE(da && db && dw && dout && dy && dq && dd && ddk);
            auto run = [&](const char* k, const auto& args) {
                return chunked ? kcLaunchKernelChunked(k, &args, workers(), 8) : kcLaunchKernel(k, &args);
            };
            CHECK_OK(run("add_sumsq_f", llm::AddSumsqFArgs{da, db, dout, dq + off, dd, n, kr}));
            CHECK_OK(run("norm_mul_q8k_f", llm::NormMulQ8Args{dout, dw, dy, dq, ddk, n, kr, 1e-5f}));
            CHECK(bitEqual(hostCopy(dy, ky.size()), ky));
            CHECK(hostCopy(dq, kq.size()) == kq);
            auto *dg = devCopy(ks), *du = devCopy(ku), *dsy = devCopy(std::vector<float>(ks.size()));
            auto* dsq = devCopy(std::vector<int>(ksq8.size()));
            auto* dsd = devCopy(std::vector<int>{m, kr});
            REQUIRE(dg && du && dsy && dsq && dsd);
            CHECK_OK(run("silu_mul_q8kf", llm::SiluMulQ8FArgs{dg, du, dsy, dsq, dsd, m, kr}));
            CHECK(bitEqual(hostCopy(dsy, ksy.size()), ksy));
            CHECK(hostCopy(dsq, ksq8.size()) == ksq8);
        }
    }
}

// Fused kernels give exactly the results of the sequences they replace.
TEST_CASE(fused_kernels_equal_unfused_sequences) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int n = 3072;
    const auto x = randomFloats(size_t(n), 41, -2, 2), t = randomFloats(size_t(n), 42, -2, 2);
    const auto w = randomFloats(size_t(n), 43, 0.9f, 1.1f);
    std::vector<float> rx = x, ry(static_cast<size_t>(n));
    llm::ref::add(rx.data(), t.data(), n);
    llm::ref::rmsNorm(rx.data(), w.data(), ry.data(), n, 1, 1e-5f);
    auto* dx = devCopy(x);
    auto* dt = devCopy(t);
    auto* dw = devCopy(w);
    auto* dy = devCopy(std::vector<float>(size_t(n), 0.0f));
    REQUIRE(dx && dt && dw && dy);
    const llm::AddRmsNormArgs ar{dx, dt, dw, dy, n, 1e-5f};
    CHECK_OK(kcLaunchKernel("add_rms_norm", &ar));
    CHECK(bitEqual(hostCopy(dx, size_t(n)), rx));
    CHECK(bitEqual(hostCopy(dy, size_t(n)), ry));

    const int count = 256;  // 8192 values, the 3B FFN width
    const auto g = randomFloats(size_t(count) * 32, 44, -6, 6), u = randomFloats(size_t(count) * 32, 45, -2, 2);
    std::vector<float> h(g.size());
    llm::ref::siluMul(g.data(), u.data(), h.data(), int(h.size()));
    std::vector<int> xq(size_t(count) * 8);
    std::vector<float> xd(size_t(count) * 2);
    llm::ref::quantizeQ8x4(h.data(), xq.data(), xd.data(), count);
    auto* dg = devCopy(g);
    auto* du = devCopy(u);
    auto* dxq = devCopy(std::vector<int>(xq.size(), 0));
    auto* dxd = devCopy(std::vector<float>(xd.size(), 0.0f));
    REQUIRE(dg && du && dxq && dxd);
    const llm::SiluMulQ8Args sq{dg, du, dxq, dxd, count};
    CHECK_OK(kcLaunchKernel("silu_mul_q8x4", &sq));
    CHECK(hostCopy(dxq, xq.size()) == xq);
    CHECK(bitEqual(hostCopy(dxd, xd.size()), xd));
    CHECK_OK(kcMemset(dxq, 0, xq.size() * 4));
    CHECK_OK(kcLaunchKernelChunked("silu_mul_q8x4", &sq, workers(), 8));
    CHECK(hostCopy(dxq, xq.size()) == xq);
}

TEST_CASE(parallel_norm_quant_matches_reference_and_chunks) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int nb = 96;  // the 3B hidden size, 3072
    const auto x = randomFloats(size_t(nb) * 32, 51, -2, 2), t = randomFloats(size_t(nb) * 32, 52, -2, 2);
    const auto w = randomFloats(size_t(nb) * 32, 53, 0.9f, 1.1f);
    std::vector<float> rx = x;
    std::vector<int> rq(size_t(llm::scratchInts(nb)), 0);
    llm::ref::addSumsq(rx.data(), t.data(), rq.data(), nb);
    llm::ref::normQ8x4(rx.data(), w.data(), rq.data(), nb, 1e-5f);
    for (bool chunked : {false, true}) {
        auto* dx = devCopy(x);
        auto* dt = devCopy(t);
        auto* dw = devCopy(w);
        auto* dq = devCopy(std::vector<int>(rq.size(), 0));
        REQUIRE(dx && dt && dw && dq);
        const llm::AddSumsqArgs s{dx, dt, dq, nb};
        const llm::NormQ8Args n{dx, dw, dq, nb, 1e-5f};
        CHECK_OK(chunked ? kcLaunchKernelChunked("add_sumsq", &s, workers(), 16) : kcLaunchKernel("add_sumsq", &s));
        CHECK_OK(chunked ? kcLaunchKernelChunked("norm_q8x4", &n, workers(), 16) : kcLaunchKernel("norm_q8x4", &n));
        CHECK(bitEqual(hostCopy(dx, rx.size()), rx));
        CHECK(hostCopy(dq, rq.size()) == rq);  // words, scales and partial sums, bit for bit
    }
    // Close to the unfused float path: quantizing the plain RMS norm gives the same bytes but for
    // rare rounding ties (the sum of squares is added in a different order).
    std::vector<float> ax = x, y(x.size());
    llm::ref::add(ax.data(), t.data(), nb * 32);
    llm::ref::rmsNorm(ax.data(), w.data(), y.data(), nb * 32, 1, 1e-5f);
    std::vector<int> q2(size_t(nb) * 8);
    std::vector<float> d2(size_t(nb) * 2);
    llm::ref::quantizeQ8x4(y.data(), q2.data(), d2.data(), nb);
    int same = 0;
    for (int i = 0; i < nb * 8; ++i) same += q2[size_t(i)] == rq[size_t(i)];
    CHECK(same >= nb * 8 * 95 / 100);
}


TEST_CASE(attention_kernels_match_reference_and_chunk) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    // Grouped-query shape: 6 query heads share 2 KV heads; ctx not a multiple of 8 or 64.
    // Scores rows are padded to 336; the transposed K cache holds 340 positions.
    const llm::AttnDims d{6, 333, 64, 2 * 64, 3, llm::padTo8(333), 340};
    const auto q = randomFloats(size_t(d.n_heads) * d.head_dim, 1);
    const auto k = randomFloats(size_t(d.kv_stride) * d.k_stride, 2);
    const auto v = randomFloats(size_t(d.ctx) * d.kv_stride, 3);
    auto* dq = devCopy(q);
    auto* dk = devCopy(k);
    auto* dv = devCopy(v);
    auto* dsc = devCopy(std::vector<float>(size_t(d.n_heads) * d.ctx_pad, 0.0f));
    auto* dout = devCopy(std::vector<float>(size_t(d.n_heads) * d.head_dim, 0.0f));
    auto* dd = devCopy(std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride});
    REQUIRE(dq && dk && dv && dsc && dout && dd);
    const float scale = 0.125f;
    // Positions t >= ctx of each padded row are not written by design: compare the rest.
    auto valid = [&](const std::vector<float>& s) {
        std::vector<float> r;
        for (int h = 0; h < d.n_heads; ++h)
            r.insert(r.end(), s.begin() + ptrdiff_t(h) * d.ctx_pad, s.begin() + ptrdiff_t(h) * d.ctx_pad + d.ctx);
        return r;
    };

    std::vector<float> rs(size_t(d.n_heads) * d.ctx_pad), ro(size_t(d.n_heads) * d.head_dim);
    llm::ref::attnScores(q.data(), k.data(), rs.data(), d, scale);
    const llm::AttnScoresArgs sa{dq, dk, dsc, dd, d, scale};
    CHECK_OK(kcLaunchKernel("attn_scores", &sa));
    const auto scores = hostCopy(dsc, rs.size());
    CHECK(bitEqual(valid(scores), valid(rs)));
    CHECK_OK(kcMemset(dsc, 0xFF, rs.size() * 4));
    CHECK_OK(kcLaunchKernelChunked("attn_scores", &sa, workers(), 64));
    CHECK(bitEqual(valid(hostCopy(dsc, rs.size())), valid(scores)));

    llm::ref::softmax(rs.data(), d.ctx, d.ctx_pad, d.n_heads);
    const llm::SoftmaxArgs sm{dsc, d.ctx, d.ctx_pad, d.n_heads};
    CHECK_OK(kcLaunchKernel("softmax", &sm));
    const auto probs = hostCopy(dsc, rs.size());
    CHECK(relDiff(valid(probs), valid(rs)) < 1e-6);
    for (int h = 0; h < d.n_heads; ++h) {
        double sum = 0;
        for (int t = 0; t < d.ctx; ++t) sum += probs[size_t(h) * d.ctx_pad + t];
        CHECK(std::abs(sum - 1.0) < 1e-5);
    }

    llm::ref::attnValues(probs.data(), v.data(), ro.data(), d);
    const llm::AttnValuesArgs va{dsc, dv, dout, dd, d};
    CHECK_OK(kcLaunchKernel("attn_values", &va));
    const auto out = hostCopy(dout, ro.size());
    CHECK(bitEqual(out, ro));
    CHECK_OK(kcMemset(dout, 0xFF, ro.size() * 4));
    CHECK_OK(kcLaunchKernelChunked("attn_values", &va, workers(), 64));
    CHECK(bitEqual(hostCopy(dout, ro.size()), out));
}

// fp16 conversions: set_rows_f16 rounds like the reference (to nearest even) across the range.
TEST_CASE(set_rows_f16_matches_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int pairs = 64, rows = 5, src_row = 2 * pairs + 6, dst_row = pairs + 8, dst_rows = 9;
    auto src = randomFloats(size_t(rows) * src_row, 91, -3, 3);
    // Special values: zero, subnormal range, a rounding tie, near the top of the range, overflow.
    const float special[] = {0.0f, -0.0f, 1e-7f, -3e-6f, 6.1e-5f, 1.0f + 1.0f / 2048, 1.0f + 3.0f / 2048, 65504.0f,
                             65519.0f, -70000.0f, 2.5e-8f, 2.9e-8f, 1e-30f};
    for (size_t i = 0; i < std::size(special); ++i) src[i] = special[i];
    for (int i = 0; i < 300; ++i) src[200 + i] *= std::ldexp(1.0f, i % 40 - 25);  // many exponents
    const std::vector<int64_t> idx{7, 0, 3, 99, 8};  // 99: outside the cache, skipped
    std::vector<uint16_t> ref(size_t(dst_rows) * dst_row * 2, 0x1234);
    llm::ref::setRowsF16(src.data(), idx.data(), ref.data(), pairs, rows, src_row, dst_row, dst_rows);
    std::vector<int> init(ref.size() / 2, 0x12341234), refw(init.size());
    std::memcpy(refw.data(), ref.data(), ref.size() * 2);
    std::vector<int> idxw(idx.size() * 2);
    std::memcpy(idxw.data(), idx.data(), idx.size() * 8);
    for (bool chunked : {false, true}) {
        auto* ds = devCopy(src);
        auto* dst = devCopy(init);
        auto *di = devCopy(idxw), *dd = devCopy(std::vector<int>{pairs, rows, src_row, dst_row, dst_rows});
        REQUIRE(ds && dst && di && dd);
        const llm::SetRowsF16Args a{ds, di, dst, dd, pairs, rows, src_row, dst_row, dst_rows};
        CHECK_OK(chunked ? kcLaunchKernelChunked("set_rows_f16", &a, workers(), 8) : kcLaunchKernel("set_rows_f16", &a));
        CHECK(hostCopy(dst, refw.size()) == refw);
    }
    // f16ToF32(f32ToF16(x)) is within half an fp16 step of x.
    for (float x : {0.3f, -1.7f, 1234.5f, 1e-6f})
        CHECK(std::abs(llm::ref::f16ToF32(llm::ref::f32ToF16(x)) - x) <= std::abs(x) / 2048 + 6e-8f);
}

// matmul_f16: fp16 weights (repacked by 8 rows) times fp32 columns, 1 to 6 columns (tiles of 4).
TEST_CASE(matmul_f16_matches_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int rows = 24, k = 96;
    std::vector<uint16_t> w(size_t(rows) * k);
    const auto wf = randomFloats(w.size(), 95, -1, 1);
    for (size_t i = 0; i < w.size(); ++i) w[i] = llm::ref::f32ToF16(wf[i]);
    std::vector<int> stream(w.size() / 2);
    llm::ref::repackF16(w.data(), rows, k, stream.data());
    std::vector<uint16_t> back(w.size());
    llm::ref::unpackF16(stream.data(), rows, k, back.data());
    CHECK(back == w);
    for (int ncols : {1, 4, 6}) {
        const auto x = randomFloats(size_t(ncols) * k, 96 + ncols, -2, 2);
        std::vector<float> ref(size_t(ncols) * rows);
        llm::ref::matmulF16(w.data(), x.data(), ref.data(), rows, k, ncols);
        for (bool chunked : {false, true}) {
            auto* dw = devCopy(stream);
            auto* dx = devCopy(x);
            auto* dy = devCopy(std::vector<float>(ref.size(), 0.0f));
            auto* dd = devCopy(std::vector<int>{rows, k, ncols});
            REQUIRE(dw && dx && dy && dd);
            const llm::MatmulF16Args a{dw, dx, dy, dd, rows, k, ncols};
            CHECK_OK(chunked ? kcLaunchKernelChunked("matmul_f16", &a, workers(), 8) : kcLaunchKernel("matmul_f16", &a));
            CHECK(relDiff(hostCopy(dy, ref.size()), ref) < 1e-6);
        }
    }
}

// fa_scores / fa_max / fa_exp / fa_values against a direct softmax attention over fp16 K and V,
// laid out as llama.cpp's KV cache (rows of H_kv * D fp16) and permuted Q.
TEST_CASE(flash_attention_kernels_match_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    struct Shape { int H, H_kv, D, T, N; bool mask; };
    for (const Shape sh : {Shape{6, 2, 64, 37, 1, false}, Shape{10, 2, 64, 37, 3, true}, Shape{8, 8, 128, 300, 2, true}}) {
        llm::FaDims d;
        d.H = sh.H, d.H_kv = sh.H_kv, d.D = sh.D, d.T = sh.T, d.N = sh.N, d.group = sh.H / sh.H_kv;
        d.tp = llm::padTo8(sh.T);
        const int kv_size = sh.T + 11;
        d.k_row = d.v_row = sh.H_kv * sh.D / 2, d.k_head = d.v_head = sh.D / 2;
        d.q_tok = sh.H * sh.D, d.q_head = sh.D;  // ggml: Q [D, N, H] viewed from [D, H, N]
        d.m_row = (sh.T + 7) / 2 + 3, d.has_mask = sh.mask ? 1 : 0;
        const auto q = randomFloats(size_t(sh.N) * d.q_tok, 92, -1, 1);
        auto f16s = [](const std::vector<float>& f) {
            std::vector<uint16_t> h(f.size());
            for (size_t i = 0; i < f.size(); ++i) h[i] = llm::ref::f32ToF16(f[i]);
            return h;
        };
        const auto k = f16s(randomFloats(size_t(kv_size) * d.k_row * 2, 93, -1, 1));
        const auto v = f16s(randomFloats(size_t(kv_size) * d.v_row * 2, 94, -1, 1));
        std::vector<uint16_t> mask(size_t(sh.N) * d.m_row * 2, 0);
        for (int i = 0; i < sh.N; ++i)  // causal-like: the last N - 1 - i positions are hidden from query i
            for (int t = sh.T - (sh.N - 1 - i); t < sh.T; ++t) mask[size_t(i) * d.m_row * 2 + t] = 0xFC00;
        const float scale = 1.0f / std::sqrt(float(sh.D));
        std::vector<float> ref(size_t(sh.N) * sh.H * sh.D);
        llm::ref::flashAttnF16(q.data(), k.data(), v.data(), mask.data(), ref.data(), d, scale);
        auto words = [](const std::vector<uint16_t>& h) {
            std::vector<int> w(h.size() / 2);
            std::memcpy(w.data(), h.data(), w.size() * 4);
            return w;
        };
        std::vector<float> first;
        for (bool chunked : {false, true}) {
            auto *dq = devCopy(q), *ds = devCopy(std::vector<float>(size_t(sh.H) * sh.N * d.tp, 0.0f));
            auto *dk = devCopy(words(k)), *dv = devCopy(words(v)), *dm = devCopy(words(mask));
            auto *dmx = devCopy(std::vector<float>(size_t(sh.H) * sh.N)), *dout = devCopy(std::vector<float>(ref.size()));
            std::vector<int> dims(llm::kFaDims);
            llm::faDimsInts(d, dims.data());
            auto* dd = devCopy(dims);
            REQUIRE(dq && ds && dk && dv && dm && dmx && dout && dd);
            auto run = [&](const char* name, const auto& a) {
                return chunked ? kcLaunchKernelChunked(name, &a, workers(), 8) : kcLaunchKernel(name, &a);
            };
            CHECK_OK(run("fa_scores", llm::FaScoresArgs{dq, dk, dm, ds, dd, d, scale}));
            CHECK_OK(run("fa_max", llm::FaMaxArgs{ds, dmx, dd, d}));
            CHECK_OK(run("fa_exp", llm::FaExpArgs{ds, dmx, dd, d}));
            CHECK_OK(run("fa_values", llm::FaValuesArgs{ds, dv, dout, dd, d}));
            const auto out = hostCopy(dout, ref.size());
            CHECK(relDiff(out, ref) < 2e-6);
            if (chunked) CHECK(bitEqual(out, first));
            else first = out;
        }
    }
}

TEST_CASE(kv_append_writes_transposed_k_and_row_major_v) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    // 3 KV heads of 32: n = 96; K cache rows hold 24 positions; appending position 13 (ctx 14).
    const llm::AttnDims d{6, 14, 32, 96, 2, 16, 24};
    const int n = d.kv_stride, pos = d.ctx - 1;
    const auto k = randomFloats(size_t(n), 21), v = randomFloats(size_t(n), 22);
    auto kt = randomFloats(size_t(n) * d.k_stride, 23), vc = randomFloats(size_t(n) * d.ctx, 24);
    auto *dk = devCopy(k), *dv = devCopy(v), *dkt = devCopy(kt), *dvc = devCopy(vc);
    auto* dd = devCopy(std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride});
    REQUIRE(dk && dv && dkt && dvc && dd);
    const llm::KvAppendArgs a{dk, dv, dkt, dvc, dd, d};
    CHECK_OK(kcLaunchKernel("kv_append", &a));
    llm::ref::appendKv(k.data(), v.data(), kt.data(), vc.data(), n, d.k_stride, pos);
    CHECK(bitEqual(hostCopy(dkt, kt.size()), kt));
    CHECK(bitEqual(hostCopy(dvc, vc.size()), vc));
    llm::AttnDims big = d;  // context past the K cache rows
    big.ctx = 25, big.ctx_pad = 32;
    const llm::KvAppendArgs past{dk, dv, dkt, dvc, dd, big};
    CHECK_ERR(kcLaunchKernel("kv_append", &past), Error::InvalidValue);
}

TEST_CASE(norm_rope_silu_match_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    const int n = 384, rows = 5;
    const auto x = randomFloats(size_t(n) * rows, 11);
    const auto w = randomFloats(size_t(n), 12, 0.9f, 1.1f);
    auto* dx = devCopy(x);
    auto* dw = devCopy(w);
    auto* dy = devCopy(std::vector<float>(x.size(), 0.0f));
    REQUIRE(dx && dw && dy);

    std::vector<float> ry(x.size());
    llm::ref::rmsNorm(x.data(), w.data(), ry.data(), n, rows, 1e-5f);
    const llm::RmsNormArgs ra{dx, dw, dy, n, rows, 1e-5f};
    CHECK_OK(kcLaunchKernel("rms_norm", &ra));
    CHECK(relDiff(hostCopy(dy, x.size()), ry) < 1e-6);
    CHECK_OK(kcLaunchKernelChunked("rms_norm", &ra, 2, 1));
    CHECK(relDiff(hostCopy(dy, x.size()), ry) < 1e-6);

    // rope: 3 heads of head_dim 128 at a large position. Booth lowers sin/cos arguments (F11),
    // so compare loosely; magnitudes are preserved exactly by a rotation up to rounding.
    const int pairs = 64, heads = 3, pos = 4321;
    std::vector<float> rx = randomFloats(size_t(heads) * pairs * 2, 13);
    auto* drx = devCopy(rx);
    REQUIRE(drx);
    const float lb = std::log2(500000.0f);
    llm::ref::rope(rx.data(), pairs, pos, heads * pairs, lb);
    const llm::RopeArgs ro{drx, pairs, pos, heads * pairs, lb};
    CHECK_OK(kcLaunchKernel("rope", &ro));
    CHECK(relDiff(hostCopy(drx, rx.size()), rx) < 1e-3);

    const int count = 5000;
    const auto g = randomFloats(size_t(count), 14, -6.0f, 6.0f);
    const auto u = randomFloats(size_t(count), 15);
    auto* dg = devCopy(g);
    auto* du = devCopy(u);
    auto* dout = devCopy(std::vector<float>(size_t(count), 0.0f));
    REQUIRE(dg && du && dout);
    std::vector<float> rout(static_cast<size_t>(count));
    llm::ref::siluMul(g.data(), u.data(), rout.data(), count);
    const llm::SiluMulArgs sa{dg, du, dout, count};
    CHECK_OK(kcLaunchKernel("silu_mul", &sa));
    CHECK(bitEqual(hostCopy(dout, size_t(count)), rout));
    CHECK_OK(kcLaunchKernelChunked("silu_mul", &sa, workers(), 64));
    CHECK(bitEqual(hostCopy(dout, size_t(count)), rout));
}

TEST_CASE(llm_argument_validation) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    auto* buf = devCopy(std::vector<float>(1024, 0.0f));
    auto* q = devCopy(std::vector<int8_t>(64 * 32, 1));
    auto* dims = devCopy(std::vector<int>{64, 32});
    REQUIRE(buf && q && dims);
    // k must be a multiple of 32.
    llm::MatvecQ8Args a{q, buf, buf, buf, dims, 64, 31};
    CHECK_ERR(kcLaunchKernel("matvec_q8", &a), Error::InvalidValue);
    // Device dims must agree with the validated sizes.
    a = {q, buf, buf, buf, dims, 32, 32};
    CHECK_ERR(kcLaunchKernel("matvec_q8", &a), Error::InvalidValue);
    // Weight buffer too small for 128 rows.
    auto* dims128 = devCopy(std::vector<int>{128, 32});
    a = {q, buf, buf, buf, dims128, 128, 32};
    CHECK_ERR(kcLaunchKernel("matvec_q8", &a), Error::InvalidPointer);
    // Heads not divisible by the group size.
    const llm::AttnDims bad{5, 4, 32, 64, 2, 8, 4};
    const llm::AttnScoresArgs s{buf, buf, buf, dims, bad, 1.0f};
    CHECK_ERR(kcLaunchKernel("attn_scores", &s), Error::InvalidValue);
    // rope count must be whole heads.
    const llm::RopeArgs r{buf, 64, 0, 100, 1.0f};
    CHECK_ERR(kcLaunchKernel("rope", &r), Error::InvalidValue);
}

TEST_CASE(tiny_decode_token_kc_vs_cpp) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (llm::Weights w : {llm::Weights::Q8, llm::Weights::Q4, llm::Weights::Q4X8, llm::Weights::Q4X8Q8}) {
        const llm::Config cfg = llm::tinyConfig(w);
        llm::KcAllocator pa;
        llm::HostAllocator ha;
        std::vector<llm::Model> models;
        llm::buildModels(cfg, 42, {&pa}, models);
        const llm::Model& m = models[0];
        const llm::Acts ap = llm::allocActs(cfg, pa, 7);
        const llm::Acts ah = llm::allocActs(cfg, ha, 7);

        auto serial = llm::makeKcBackend(m, 0, false);
        auto chunked = llm::makeKcBackend(m, workers(), false);
        auto graph = llm::makeKcBackend(m, workers(), true);  // each token as one Kernelcore graph
        auto two_rows = llm::makeKcBackend(m, workers(), true, true);  // matvec_q4x8_q8x2 where rows % 16 == 0
        auto cpp = llm::makeCppBackend(m);

        // Same position each time: every run appends its K/V at slot 17, then attends over 0..17.
        llm::decodeToken(*serial, m, ap, 17);
        const auto ls = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*chunked, m, ap, 17);
        const auto lc = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*graph, m, ap, 17);
        const auto lg = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*two_rows, m, ap, 17);
        const auto l2 = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*cpp, m, ah, 17);
        const std::vector<float> lr(ah.logits, ah.logits + cfg.vocab);

        CHECK(bitEqual(ls, lc));
        CHECK(bitEqual(ls, lg));
        CHECK(bitEqual(ls, l2));
        const double d = relDiff(ls, lr);  // only rope's sin/cos lowering differs
        if (!(d < 1e-4)) th::fail(__FILE__, __LINE__, "logits differ from C++ reference: " + std::to_string(d));
        CHECK(std::max_element(ls.begin(), ls.end()) - ls.begin() ==
              std::max_element(lr.begin(), lr.end()) - lr.begin());
    }
}
