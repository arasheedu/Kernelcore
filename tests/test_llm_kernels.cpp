// LLM decode kernels (kernels/llm/*.cu) under pcuda: each against the plain C++ reference,
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

using pcuda::CopyKind;
using pcuda::Error;

namespace {

struct LlmFixture {
    RuntimeFixture rt;
    bool ok = false;
    LlmFixture() { ok = rt.ok && llm::registerKernels() == Error::Success; }
};

template <class T>
T* devCopy(const std::vector<T>& h) {
    void* p = nullptr;
    if (pcudaMalloc(&p, h.size() * sizeof(T)) != Error::Success) return nullptr;
    if (pcudaMemcpy(p, h.data(), h.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}

template <class T>
std::vector<T> hostCopy(const T* d, size_t n) {
    std::vector<T> h(n);
    if (pcudaMemcpy(h.data(), d, n * sizeof(T), CopyKind::DeviceToHost) != Error::Success) h.clear();
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
        CHECK_OK(pcudaLaunchKernel("matvec_q8", &a));
        const auto serial = hostCopy(dy, size_t(rows));
        CHECK(bitEqual(serial, ref));  // same loop order, no FMA: bit-exact
        CHECK_OK(pcudaMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(pcudaLaunchKernelChunked("matvec_q8", &a, workers(), 64));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), serial));
        for (void* p : {static_cast<void*>(dq), static_cast<void*>(ds), static_cast<void*>(dx),
                        static_cast<void*>(dy), static_cast<void*>(dd)})
            CHECK_OK(pcudaFree(p));
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
        CHECK_OK(pcudaLaunchKernel("matvec_q4x8", &a));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want8));
        CHECK_OK(pcudaMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(pcudaLaunchKernelChunked("matvec_q4x8", &a, workers(), 8));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want8));
    }
    // rows must be a multiple of 8
    auto* buf = devCopy(std::vector<float>(1024, 0.0f));
    auto* q = devCopy(std::vector<uint8_t>(4096, 0));
    auto* dims = devCopy(std::vector<int>{12, 32});
    REQUIRE(buf && q && dims);
    const llm::MatvecQ4Args bad{q, buf, buf, buf, dims, 12, 32};
    CHECK_ERR(pcudaLaunchKernel("matvec_q4x8", &bad), Error::InvalidValue);
}

// 8-bit activations + integer dot products: exact against the C++ reference, and within
// quantization error of the float kernel on the same weights.
TEST_CASE(matvec_q4x8_q8_matches_reference) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (auto [rows, k] : {std::pair{8, 32}, std::pair{96, 64}, std::pair{1000, 512}}) {
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
        CHECK_OK(pcudaLaunchKernel("quantize_q8x4", &qa));
        CHECK(hostCopy(dxq, xq.size()) == xq);
        CHECK(bitEqual(hostCopy(dxd, xd.size()), xd));
        const llm::MatvecQ4Q8Args a{dqw, dxq, dxd, dy, dd, rows, k};
        CHECK_OK(pcudaLaunchKernel("matvec_q4x8_q8", &a));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
        CHECK_OK(pcudaMemset(dy, 0xFF, size_t(rows) * 4));
        CHECK_OK(pcudaLaunchKernelChunked("matvec_q4x8_q8", &a, workers(), 8));
        CHECK(bitEqual(hostCopy(dy, size_t(rows)), want));
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
    CHECK_OK(pcudaLaunchKernel("add_rms_norm", &ar));
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
    CHECK_OK(pcudaLaunchKernel("silu_mul_q8x4", &sq));
    CHECK(hostCopy(dxq, xq.size()) == xq);
    CHECK(bitEqual(hostCopy(dxd, xd.size()), xd));
    CHECK_OK(pcudaMemset(dxq, 0, xq.size() * 4));
    CHECK_OK(pcudaLaunchKernelChunked("silu_mul_q8x4", &sq, workers(), 8));
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
        CHECK_OK(chunked ? pcudaLaunchKernelChunked("add_sumsq", &s, workers(), 16) : pcudaLaunchKernel("add_sumsq", &s));
        CHECK_OK(chunked ? pcudaLaunchKernelChunked("norm_q8x4", &n, workers(), 16) : pcudaLaunchKernel("norm_q8x4", &n));
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
    CHECK_OK(pcudaLaunchKernel("attn_scores", &sa));
    const auto scores = hostCopy(dsc, rs.size());
    CHECK(bitEqual(valid(scores), valid(rs)));
    CHECK_OK(pcudaMemset(dsc, 0xFF, rs.size() * 4));
    CHECK_OK(pcudaLaunchKernelChunked("attn_scores", &sa, workers(), 64));
    CHECK(bitEqual(valid(hostCopy(dsc, rs.size())), valid(scores)));

    llm::ref::softmax(rs.data(), d.ctx, d.ctx_pad, d.n_heads);
    const llm::SoftmaxArgs sm{dsc, d.ctx, d.ctx_pad, d.n_heads};
    CHECK_OK(pcudaLaunchKernel("softmax", &sm));
    const auto probs = hostCopy(dsc, rs.size());
    CHECK(relDiff(valid(probs), valid(rs)) < 1e-6);
    for (int h = 0; h < d.n_heads; ++h) {
        double sum = 0;
        for (int t = 0; t < d.ctx; ++t) sum += probs[size_t(h) * d.ctx_pad + t];
        CHECK(std::abs(sum - 1.0) < 1e-5);
    }

    llm::ref::attnValues(probs.data(), v.data(), ro.data(), d);
    const llm::AttnValuesArgs va{dsc, dv, dout, dd, d};
    CHECK_OK(pcudaLaunchKernel("attn_values", &va));
    const auto out = hostCopy(dout, ro.size());
    CHECK(bitEqual(out, ro));
    CHECK_OK(pcudaMemset(dout, 0xFF, ro.size() * 4));
    CHECK_OK(pcudaLaunchKernelChunked("attn_values", &va, workers(), 64));
    CHECK(bitEqual(hostCopy(dout, ro.size()), out));
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
    CHECK_OK(pcudaLaunchKernel("kv_append", &a));
    llm::ref::appendKv(k.data(), v.data(), kt.data(), vc.data(), n, d.k_stride, pos);
    CHECK(bitEqual(hostCopy(dkt, kt.size()), kt));
    CHECK(bitEqual(hostCopy(dvc, vc.size()), vc));
    llm::AttnDims big = d;  // context past the K cache rows
    big.ctx = 25, big.ctx_pad = 32;
    const llm::KvAppendArgs past{dk, dv, dkt, dvc, dd, big};
    CHECK_ERR(pcudaLaunchKernel("kv_append", &past), Error::InvalidValue);
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
    CHECK_OK(pcudaLaunchKernel("rms_norm", &ra));
    CHECK(relDiff(hostCopy(dy, x.size()), ry) < 1e-6);
    CHECK_OK(pcudaLaunchKernelChunked("rms_norm", &ra, 2, 1));
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
    CHECK_OK(pcudaLaunchKernel("rope", &ro));
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
    CHECK_OK(pcudaLaunchKernel("silu_mul", &sa));
    CHECK(bitEqual(hostCopy(dout, size_t(count)), rout));
    CHECK_OK(pcudaLaunchKernelChunked("silu_mul", &sa, workers(), 64));
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
    CHECK_ERR(pcudaLaunchKernel("matvec_q8", &a), Error::InvalidValue);
    // Device dims must agree with the validated sizes.
    a = {q, buf, buf, buf, dims, 32, 32};
    CHECK_ERR(pcudaLaunchKernel("matvec_q8", &a), Error::InvalidValue);
    // Weight buffer too small for 128 rows.
    auto* dims128 = devCopy(std::vector<int>{128, 32});
    a = {q, buf, buf, buf, dims128, 128, 32};
    CHECK_ERR(pcudaLaunchKernel("matvec_q8", &a), Error::InvalidPointer);
    // Heads not divisible by the group size.
    const llm::AttnDims bad{5, 4, 32, 64, 2, 8, 4};
    const llm::AttnScoresArgs s{buf, buf, buf, dims, bad, 1.0f};
    CHECK_ERR(pcudaLaunchKernel("attn_scores", &s), Error::InvalidValue);
    // rope count must be whole heads.
    const llm::RopeArgs r{buf, 64, 0, 100, 1.0f};
    CHECK_ERR(pcudaLaunchKernel("rope", &r), Error::InvalidValue);
}

TEST_CASE(tiny_decode_token_pcuda_vs_cpp) {
    LlmFixture fx;
    REQUIRE(fx.ok);
    for (llm::Weights w : {llm::Weights::Q8, llm::Weights::Q4, llm::Weights::Q4X8, llm::Weights::Q4X8Q8}) {
        const llm::Config cfg = llm::tinyConfig(w);
        llm::PcudaAllocator pa;
        llm::HostAllocator ha;
        std::vector<llm::Model> models;
        llm::buildModels(cfg, 42, {&pa}, models);
        const llm::Model& m = models[0];
        const llm::Acts ap = llm::allocActs(cfg, pa, 7);
        const llm::Acts ah = llm::allocActs(cfg, ha, 7);

        auto serial = llm::makePcudaBackend(m, 0, false);
        auto chunked = llm::makePcudaBackend(m, workers(), false);
        auto graph = llm::makePcudaBackend(m, workers(), true);  // each token as one pcuda graph
        auto cpp = llm::makeCppBackend(m);

        // Same position each time: every run appends its K/V at slot 17, then attends over 0..17.
        llm::decodeToken(*serial, m, ap, 17);
        const auto ls = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*chunked, m, ap, 17);
        const auto lc = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*graph, m, ap, 17);
        const auto lg = hostCopy(ap.logits, size_t(cfg.vocab));
        llm::decodeToken(*cpp, m, ah, 17);
        const std::vector<float> lr(ah.logits, ah.logits + cfg.vocab);

        CHECK(bitEqual(ls, lc));
        CHECK(bitEqual(ls, lg));
        const double d = relDiff(ls, lr);  // only rope's sin/cos lowering differs
        if (!(d < 1e-4)) th::fail(__FILE__, __LINE__, "logits differ from C++ reference: " + std::to_string(d));
        CHECK(std::max_element(ls.begin(), ls.end()) - ls.begin() ==
              std::max_element(lr.begin(), lr.end()) - lr.begin());
    }
}
