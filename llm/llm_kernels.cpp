// pcuda adapters for kernels/llm/*.cu. Each validates every buffer against live pcuda
// allocations before the Booth code runs, and packs System V arguments in declaration order.
// Kernel indices are 32-bit ints, so every element count must fit in int32.

#include "llm_kernels.hpp"

#include <pcuda/kernel_registry.hpp>

#include <limits>

PCUDA_DECLARE_KERNEL_IMAGE(matvec_q8);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q4);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q4x8);
PCUDA_DECLARE_KERNEL_IMAGE(quantize_q8x4);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q4x8_q8);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q4x8_q8x2);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q4k_q8);
PCUDA_DECLARE_KERNEL_IMAGE(matvec_q6k_q8);
PCUDA_DECLARE_KERNEL_IMAGE(rope_ff);
PCUDA_DECLARE_KERNEL_IMAGE(add_rms_norm);
PCUDA_DECLARE_KERNEL_IMAGE(silu_mul_q8x4);
PCUDA_DECLARE_KERNEL_IMAGE(add_sumsq);
PCUDA_DECLARE_KERNEL_IMAGE(norm_q8x4);
PCUDA_DECLARE_KERNEL_IMAGE(rms_norm);
PCUDA_DECLARE_KERNEL_IMAGE(rope);
PCUDA_DECLARE_KERNEL_IMAGE(kv_append);
PCUDA_DECLARE_KERNEL_IMAGE(attn_scores);
PCUDA_DECLARE_KERNEL_IMAGE(softmax);
PCUDA_DECLARE_KERNEL_IMAGE(attn_values);
PCUDA_DECLARE_KERNEL_IMAGE(silu_mul);

namespace llm {
namespace {

using pcuda::ArgPacker;
using pcuda::Error;
using pcuda::LaunchArgs;
using pcuda::LaunchContext;

constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();

bool fitsIndex(int64_t a, int64_t b = 1) { return a >= 0 && b >= 0 && (b == 0 || a <= kMaxIndex / b); }

size_t f32Bytes(int64_t n) { return static_cast<size_t>(n) * sizeof(float); }

Error check(const LaunchContext& ctx, const void* p, size_t bytes, const char* what) {
    return ctx.checkDeviceRange(p, bytes, what);
}

#define LLM_TRY(expr)                                   \
    do {                                                \
        if (Error e_ = (expr); e_ != Error::Success) return e_; \
    } while (0)

// Chunk ranges handed out by the runtime's planner.
bool chunkOk(int64_t total, int64_t begin, int64_t length) {
    return begin >= 0 && length > 0 && begin + length <= total && length <= kMaxIndex;
}

// pcuda device memory is host memory, so the adapter can confirm that the device-side dims
// array agrees with the sizes it validated. A mismatch would let the kernel index past a buffer.
bool dimsMatch(const int* dims, const AttnDims& d) {
    return dims[0] == d.n_heads && dims[1] == d.ctx && dims[2] == d.head_dim && dims[3] == d.kv_stride &&
           dims[4] == d.group && dims[5] == d.ctx_pad && dims[6] == d.k_stride;
}

Error checkAttnShape(const AttnDims& d) {
    if (d.n_heads <= 0 || d.ctx <= 0 || d.head_dim <= 0 || d.group <= 0 || d.n_heads % d.group != 0)
        return Error::InvalidValue;
    if (!fitsIndex(d.n_heads, d.ctx) || !fitsIndex(d.n_heads, d.head_dim) || !fitsIndex(d.ctx, d.kv_stride))
        return Error::InvalidValue;
    if (static_cast<int64_t>(d.n_heads / d.group) * d.head_dim > d.kv_stride) return Error::InvalidValue;
    // The kernels promise pcuda_aligned8() for head_dim and ctx_pad.
    if (d.head_dim % 8 || d.ctx_pad % 8 || d.ctx_pad < d.ctx || d.k_stride < d.ctx) return Error::InvalidValue;
    if (!fitsIndex(d.n_heads, d.ctx_pad) || !fitsIndex(d.kv_stride, d.k_stride)) return Error::InvalidValue;
    return Error::Success;
}

// ---- matvec_q8(q, scales, x, y, dims) -----------------------------------------

Error matvecMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ8Args*>(p);
    if (a.rows < 0 || a.k <= 0 || a.k % 32 != 0 || !fitsIndex(a.rows, a.k)) return Error::InvalidValue;
    const int64_t elems = int64_t(a.rows) * a.k;
    LLM_TRY(check(ctx, a.q, static_cast<size_t>(elems), "q"));
    LLM_TRY(check(ctx, a.scales, f32Bytes(elems / 32), "scales"));
    LLM_TRY(check(ctx, a.x, f32Bytes(a.k), "x"));
    LLM_TRY(check(ctx, a.y, f32Bytes(a.rows), "y"));
    LLM_TRY(check(ctx, a.dims, 2 * sizeof(int), "dims"));
    if (a.dims[0] != a.rows || a.dims[1] != a.k) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.q).ptr(a.scales).ptr(a.x).ptr(a.y).ptr(a.dims);
    return pk.finish(a.rows);
}

Error matvecExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const MatvecQ8Args*>(p)->rows;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

// Rows [begin, begin + length): offset W, scales and y. dims keeps the full row count, which
// only bounds the local row index from above.
Error matvecChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ8Args*>(p);
    if (!chunkOk(a.rows, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.q + begin * a.k).ptr(a.scales + begin * (a.k / 32)).ptr(a.x).ptr(a.y + begin).ptr(a.dims);
    return pk.finish(length);
}

// ---- matvec_q4(qs, scales, x, y, dims) ----------------------------------------

Error matvecQ4Marshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Args*>(p);
    if (a.rows < 0 || a.k <= 0 || a.k % 32 != 0 || !fitsIndex(a.rows, a.k)) return Error::InvalidValue;
    const int64_t elems = int64_t(a.rows) * a.k;
    LLM_TRY(check(ctx, a.qs, static_cast<size_t>(elems / 2), "qs"));
    LLM_TRY(check(ctx, a.scales, f32Bytes(elems / 32), "scales"));
    LLM_TRY(check(ctx, a.x, f32Bytes(a.k), "x"));
    LLM_TRY(check(ctx, a.y, f32Bytes(a.rows), "y"));
    LLM_TRY(check(ctx, a.dims, 2 * sizeof(int), "dims"));
    if (a.dims[0] != a.rows || a.dims[1] != a.k) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qs).ptr(a.scales).ptr(a.x).ptr(a.y).ptr(a.dims);
    return pk.finish(a.rows);
}

Error matvecQ4Extent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const MatvecQ4Args*>(p)->rows;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error matvecQ4Chunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Args*>(p);
    if (!chunkOk(a.rows, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qs + begin * (a.k / 2)).ptr(a.scales + begin * (a.k / 32)).ptr(a.x).ptr(a.y + begin).ptr(a.dims);
    return pk.finish(length);
}

// ---- matvec_q4x8: same arguments and byte counts as matvec_q4, 8-row interleaved layout ----

Error matvecQ4x8Marshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    if (static_cast<const MatvecQ4Args*>(p)->rows % 8 != 0) return Error::InvalidValue;
    return matvecQ4Marshal(p, ctx, out);
}

// Chunks must start on an 8-row group so each thread's position within its group is unchanged.
Error matvecQ4x8Chunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    if (begin % 8 != 0) return Error::InvalidValue;
    return matvecQ4Chunk(p, begin, length, out);
}

// ---- quantize_q8x4(x, xq, xd, count) --------------------------------------------

Error quantMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const QuantizeArgs*>(p);
    if (a.count < 0 || !fitsIndex(a.count, 32)) return Error::InvalidValue;
    LLM_TRY(check(ctx, a.x, f32Bytes(int64_t(a.count) * 32), "x"));
    LLM_TRY(check(ctx, a.xq, size_t(a.count) * 8 * sizeof(int), "xq"));
    LLM_TRY(check(ctx, a.xd, f32Bytes(int64_t(a.count) * 2), "xd"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.xq).ptr(a.xd).i32(a.count);
    return pk.finish(a.count);
}

Error quantExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const QuantizeArgs*>(p)->count;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error quantChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const QuantizeArgs*>(p);
    if (!chunkOk(a.count, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x + begin * 32).ptr(a.xq + begin * 8).ptr(a.xd + begin * 2).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

// ---- matvec_q4x8_q8(qw, xq, xd, y, dims) -----------------------------------------

Error q4q8Marshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (a.rows < 0 || a.rows % 8 != 0 || a.k <= 0 || a.k % 32 != 0 || !fitsIndex(a.rows, a.k)) return Error::InvalidValue;
    const int64_t nb = a.k / 32;
    LLM_TRY(check(ctx, a.qw, size_t(int64_t(a.rows) / 8 * nb * 40 * 4), "qw"));
    LLM_TRY(check(ctx, a.xq, size_t(nb * 8 * 4), "xq"));
    LLM_TRY(check(ctx, a.xd, f32Bytes(nb * 2), "xd"));
    LLM_TRY(check(ctx, a.y, f32Bytes(a.rows), "y"));
    LLM_TRY(check(ctx, a.dims, 2 * sizeof(int), "dims"));
    if (a.dims[0] != a.rows || a.dims[1] != a.k) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qw).ptr(a.xq).ptr(a.xd).ptr(a.y).ptr(a.dims);
    return pk.finish(a.rows);
}

Error q4q8Extent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const MatvecQ4Q8Args*>(p)->rows;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

// Chunks start on an 8-row group; the weight stream advances 40 ints per group and block.
Error q4q8Chunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (!chunkOk(a.rows, begin, length) || begin % 8 != 0) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qw + begin / 8 * (a.k / 32) * 40).ptr(a.xq).ptr(a.xd).ptr(a.y + begin).ptr(a.dims);
    return pk.finish(length);
}

// ---- matvec_q4x8_q8x2: the same arguments, two rows per thread (rows % 16 == 0) ----

Error q4q8x2Marshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (a.rows % 16 != 0) return Error::InvalidValue;
    LaunchArgs checked;
    LLM_TRY(q4q8Marshal(p, ctx, &checked));  // same buffers and sizes
    ArgPacker pk(*out);
    pk.ptr(a.qw).ptr(a.xq).ptr(a.xd).ptr(a.y).ptr(a.dims);
    return pk.finish(a.rows / 2);
}

Error q4q8x2Extent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const MatvecQ4Q8Args*>(p)->rows / 2;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

// Thread t covers groups 2(t/8) and 2(t/8)+1: chunks start on a group pair (8 threads).
Error q4q8x2Chunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (!chunkOk(a.rows / 2, begin, length) || begin % 8 != 0) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qw + begin / 8 * 2 * (a.k / 32) * 40).ptr(a.xq).ptr(a.xd).ptr(a.y + begin * 2).ptr(a.dims);
    return pk.finish(length);
}

// ---- matvec_q4k_q8 / matvec_q6k_q8 (GGUF K-quant streams), same arguments ---------

template <int kWords>  // 32-bit words per row and super-block: 37 (Q4_K) or 53 (Q6_K)
Error kquantMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (a.rows < 0 || a.rows % 8 != 0 || a.k <= 0 || a.k % 256 != 0 || !fitsIndex(a.rows, a.k)) return Error::InvalidValue;
    const int64_t nb = a.k / 32, nsb = a.k / 256;
    if (!fitsIndex(int64_t(a.rows) / 8 * nsb, kWords * 8)) return Error::InvalidValue;
    LLM_TRY(check(ctx, a.qw, size_t(int64_t(a.rows) / 8 * nsb * kWords * 8 * 4), "qw"));
    LLM_TRY(check(ctx, a.xq, size_t(nb * 8 * 4), "xq"));
    LLM_TRY(check(ctx, a.xd, f32Bytes(nb * 2), "xd"));
    LLM_TRY(check(ctx, a.y, f32Bytes(a.rows), "y"));
    LLM_TRY(check(ctx, a.dims, 2 * sizeof(int), "dims"));
    if (a.dims[0] != a.rows || a.dims[1] != a.k) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qw).ptr(a.xq).ptr(a.xd).ptr(a.y).ptr(a.dims);
    return pk.finish(a.rows);
}

template <int kWords>
Error kquantChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const MatvecQ4Q8Args*>(p);
    if (!chunkOk(a.rows, begin, length) || begin % 8 != 0) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.qw + begin / 8 * (a.k / 256) * kWords * 8).ptr(a.xq).ptr(a.xd).ptr(a.y + begin).ptr(a.dims);
    return pk.finish(length);
}

// ---- rope_ff(x, ff, pairs_per_head, pos, count, log2_base) ------------------------

Error ropeFFMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const RopeFFArgs*>(p);
    if (a.pairs_per_head <= 0 || a.count < 0 || a.count % a.pairs_per_head != 0 || !fitsIndex(a.count, 2))
        return Error::InvalidValue;
    LLM_TRY(check(ctx, a.x, f32Bytes(int64_t(a.count) * 2), "x"));
    LLM_TRY(check(ctx, a.ff, f32Bytes(a.pairs_per_head), "ff"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.ff).i32(a.pairs_per_head).i32(a.pos).i32(a.count).f32(a.log2_base);
    return pk.finish(a.count);
}

// ---- add_rms_norm(x, t, w, y, n, eps): one row, one thread ------------------------

Error addRmsMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AddRmsNormArgs*>(p);
    if (a.n <= 0) return Error::InvalidValue;
    const size_t bytes = f32Bytes(a.n);
    LLM_TRY(check(ctx, a.x, bytes, "x"));
    LLM_TRY(check(ctx, a.t, bytes, "t"));
    LLM_TRY(check(ctx, a.w, bytes, "w"));
    LLM_TRY(check(ctx, a.y, bytes, "y"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.t).ptr(a.w).ptr(a.y).i32(a.n).f32(a.eps);
    return pk.finish(1);
}

// ---- silu_mul_q8x4(g, u, xq, xd, count) ------------------------------------------

Error siluQMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SiluMulQ8Args*>(p);
    if (a.count < 0 || !fitsIndex(a.count, 32)) return Error::InvalidValue;
    const size_t bytes = f32Bytes(int64_t(a.count) * 32);
    LLM_TRY(check(ctx, a.g, bytes, "g"));
    LLM_TRY(check(ctx, a.u, bytes, "u"));
    LLM_TRY(check(ctx, a.xq, size_t(a.count) * 8 * sizeof(int), "xq"));
    LLM_TRY(check(ctx, a.xd, f32Bytes(int64_t(a.count) * 2), "xd"));
    ArgPacker pk(*out);
    pk.ptr(a.g).ptr(a.u).ptr(a.xq).ptr(a.xd).i32(a.count);
    return pk.finish(a.count);
}

Error siluQExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const SiluMulQ8Args*>(p)->count;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error siluQChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SiluMulQ8Args*>(p);
    if (!chunkOk(a.count, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.g + begin * 32).ptr(a.u + begin * 32).ptr(a.xq + begin * 8).ptr(a.xd + begin * 2)
        .i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

// ---- rms_norm(x, w, y, n, rows, eps) ------------------------------------------

Error rmsMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const RmsNormArgs*>(p);
    if (a.n <= 0 || a.rows < 0 || !fitsIndex(a.rows, a.n)) return Error::InvalidValue;
    const size_t bytes = f32Bytes(int64_t(a.rows) * a.n);
    LLM_TRY(check(ctx, a.x, bytes, "x"));
    LLM_TRY(check(ctx, a.w, f32Bytes(a.n), "w"));
    LLM_TRY(check(ctx, a.y, bytes, "y"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.w).ptr(a.y).i32(a.n).i32(a.rows).f32(a.eps);
    return pk.finish(a.rows);
}

Error rmsExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const RmsNormArgs*>(p)->rows;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error rmsChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const RmsNormArgs*>(p);
    if (!chunkOk(a.rows, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x + begin * a.n).ptr(a.w).ptr(a.y + begin * a.n).i32(a.n).i32(static_cast<int32_t>(length)).f32(a.eps);
    return pk.finish(length);
}

// ---- rope(x, pairs_per_head, pos, count, log2_base) ----------------------------

Error ropeMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const RopeArgs*>(p);
    if (a.pairs_per_head <= 0 || a.count < 0 || a.count % a.pairs_per_head != 0 || !fitsIndex(a.count, 2))
        return Error::InvalidValue;
    LLM_TRY(check(ctx, a.x, f32Bytes(int64_t(a.count) * 2), "x"));
    ArgPacker pk(*out);
    pk.ptr(a.x).i32(a.pairs_per_head).i32(a.pos).i32(a.count).f32(a.log2_base);
    return pk.finish(a.count);
}

// ---- attn_scores(q, k, scores, dims, base, scale) ------------------------------

Error scoresMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AttnScoresArgs*>(p);
    LLM_TRY(checkAttnShape(a.d));
    LLM_TRY(check(ctx, a.q, f32Bytes(int64_t(a.d.n_heads) * a.d.head_dim), "q"));
    LLM_TRY(check(ctx, a.k, f32Bytes(int64_t(a.d.kv_stride) * a.d.k_stride), "k"));
    LLM_TRY(check(ctx, a.scores, f32Bytes(int64_t(a.d.n_heads) * a.d.ctx_pad), "scores"));
    LLM_TRY(check(ctx, a.dims, 7 * sizeof(int), "dims"));
    if (!dimsMatch(a.dims, a.d)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.q).ptr(a.k).ptr(a.scores).ptr(a.dims).i32(0).f32(a.scale);
    return pk.finish(int64_t(a.d.n_heads) * a.d.ctx_pad);
}

Error scoresExtent(const void* p, int64_t* total) noexcept {
    const auto& a = *static_cast<const AttnScoresArgs*>(p);
    *total = int64_t(a.d.n_heads) * a.d.ctx_pad;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

// Thread index -> (head, position) is not a pointer offset, so the chunk start goes in `base`
// (a multiple of 8: the kernel promises pcuda_aligned8(base)).
Error scoresChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AttnScoresArgs*>(p);
    if (!chunkOk(int64_t(a.d.n_heads) * a.d.ctx_pad, begin, length) || begin % 8) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.q).ptr(a.k).ptr(a.scores).ptr(a.dims).i32(static_cast<int32_t>(begin)).f32(a.scale);
    return pk.finish(length);
}

// ---- softmax(x, n, stride, rows) -----------------------------------------------

Error softmaxMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SoftmaxArgs*>(p);
    if (a.n <= 0 || a.stride < a.n || a.rows < 0 || !fitsIndex(a.rows, a.stride)) return Error::InvalidValue;
    LLM_TRY(check(ctx, a.x, f32Bytes(int64_t(a.rows) * a.stride), "x"));
    ArgPacker pk(*out);
    pk.ptr(a.x).i32(a.n).i32(a.stride).i32(a.rows);
    return pk.finish(a.rows);
}

Error softmaxExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const SoftmaxArgs*>(p)->rows;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error softmaxChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SoftmaxArgs*>(p);
    if (!chunkOk(a.rows, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x + begin * a.stride).i32(a.n).i32(a.stride).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

// ---- attn_values(p, v, out, dims, base) ----------------------------------------

Error valuesMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AttnValuesArgs*>(p);
    LLM_TRY(checkAttnShape(a.d));
    LLM_TRY(check(ctx, a.p, f32Bytes(int64_t(a.d.n_heads) * a.d.ctx_pad), "p"));
    LLM_TRY(check(ctx, a.v, f32Bytes(int64_t(a.d.ctx) * a.d.kv_stride), "v"));
    LLM_TRY(check(ctx, a.out, f32Bytes(int64_t(a.d.n_heads) * a.d.head_dim), "out"));
    LLM_TRY(check(ctx, a.dims, 7 * sizeof(int), "dims"));
    if (!dimsMatch(a.dims, a.d)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.p).ptr(a.v).ptr(a.out).ptr(a.dims).i32(0);
    return pk.finish(int64_t(a.d.n_heads) * a.d.head_dim);
}

Error valuesExtent(const void* p, int64_t* total) noexcept {
    const auto& a = *static_cast<const AttnValuesArgs*>(p);
    *total = int64_t(a.d.n_heads) * a.d.head_dim;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error valuesChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AttnValuesArgs*>(p);
    if (!chunkOk(int64_t(a.d.n_heads) * a.d.head_dim, begin, length) || begin % 8) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.p).ptr(a.v).ptr(a.out).ptr(a.dims).i32(static_cast<int32_t>(begin));
    return pk.finish(length);
}

// ---- kv_append(k, v, kt, vc, dims) ----------------------------------------------

Error kvMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const KvAppendArgs*>(p);
    LLM_TRY(checkAttnShape(a.d));
    LLM_TRY(check(ctx, a.k, f32Bytes(a.d.kv_stride), "k"));
    LLM_TRY(check(ctx, a.v, f32Bytes(a.d.kv_stride), "v"));
    LLM_TRY(check(ctx, a.kt, f32Bytes(int64_t(a.d.kv_stride) * a.d.k_stride), "kt"));
    LLM_TRY(check(ctx, a.vc, f32Bytes(int64_t(a.d.ctx) * a.d.kv_stride), "vc"));
    LLM_TRY(check(ctx, a.dims, 7 * sizeof(int), "dims"));
    if (!dimsMatch(a.dims, a.d)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.k).ptr(a.v).ptr(a.kt).ptr(a.vc).ptr(a.dims);
    return pk.finish(a.d.kv_stride);
}

// ---- add_sumsq(x, t, q, nb, base) / norm_q8x4(x, w, q, nb, base, eps) ------------

Error checkScratch(const LaunchContext& ctx, const float* a, const float* b, const int* q, int nb, const char* bn) {
    if (nb <= 0 || !fitsIndex(nb, 32)) return Error::InvalidValue;
    LLM_TRY(check(ctx, a, f32Bytes(int64_t(nb) * 32), "x"));
    LLM_TRY(check(ctx, b, f32Bytes(int64_t(nb) * 32), bn));
    return check(ctx, q, size_t(scratchInts(nb)) * sizeof(int), "q");
}

Error sumsqMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AddSumsqArgs*>(p);
    LLM_TRY(checkScratch(ctx, a.x, a.t, a.q, a.nb, "t"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.t).ptr(a.q).i32(a.nb).i32(0);
    return pk.finish(a.nb);
}
Error sumsqExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const AddSumsqArgs*>(p)->nb;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}
// Blocks are found from `base` (the scratch regions have different strides), not by offsetting.
Error sumsqChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const AddSumsqArgs*>(p);
    if (!chunkOk(a.nb, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.t).ptr(a.q).i32(a.nb).i32(static_cast<int32_t>(begin));
    return pk.finish(length);
}

Error normQMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const NormQ8Args*>(p);
    LLM_TRY(checkScratch(ctx, a.x, a.w, a.q, a.nb, "w"));
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.w).ptr(a.q).i32(a.nb).i32(0).f32(a.eps);
    return pk.finish(a.nb);
}
Error normQExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const NormQ8Args*>(p)->nb;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}
Error normQChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const NormQ8Args*>(p);
    if (!chunkOk(a.nb, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.w).ptr(a.q).i32(a.nb).i32(static_cast<int32_t>(begin)).f32(a.eps);
    return pk.finish(length);
}

// ---- silu_mul(g, u, out, count) ------------------------------------------------

Error siluMarshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SiluMulArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    const size_t bytes = f32Bytes(a.count);
    LLM_TRY(check(ctx, a.g, bytes, "g"));
    LLM_TRY(check(ctx, a.u, bytes, "u"));
    LLM_TRY(check(ctx, a.out, bytes, "out"));
    ArgPacker pk(*out);
    pk.ptr(a.g).ptr(a.u).ptr(a.out).i32(a.count);
    return pk.finish(a.count);
}

Error siluExtent(const void* p, int64_t* total) noexcept {
    *total = static_cast<const SiluMulArgs*>(p)->count;
    return *total >= 0 ? Error::Success : Error::InvalidValue;
}

Error siluChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const SiluMulArgs*>(p);
    if (!chunkOk(a.count, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.g + begin).ptr(a.u + begin).ptr(a.out + begin).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

#undef LLM_TRY

const pcuda::KernelEntry kEntries[] = {
    {"matvec_q8", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q8), &matvecMarshal, &matvecExtent, &matvecChunk},
    {"matvec_q4", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q4), &matvecQ4Marshal, &matvecQ4Extent, &matvecQ4Chunk},
    {"matvec_q4x8", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q4x8), &matvecQ4x8Marshal, &matvecQ4Extent, &matvecQ4x8Chunk},
    {"quantize_q8x4", PCUDA_KERNEL_IMAGE_FIELDS(quantize_q8x4), &quantMarshal, &quantExtent, &quantChunk},
    {"matvec_q4x8_q8", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q4x8_q8), &q4q8Marshal, &q4q8Extent, &q4q8Chunk},
    {"matvec_q4x8_q8x2", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q4x8_q8x2), &q4q8x2Marshal, &q4q8x2Extent, &q4q8x2Chunk},
    {"matvec_q4k_q8", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q4k_q8), &kquantMarshal<37>, &q4q8Extent, &kquantChunk<37>},
    {"matvec_q6k_q8", PCUDA_KERNEL_IMAGE_FIELDS(matvec_q6k_q8), &kquantMarshal<53>, &q4q8Extent, &kquantChunk<53>},
    {"rope_ff", PCUDA_KERNEL_IMAGE_FIELDS(rope_ff), &ropeFFMarshal},
    {"add_rms_norm", PCUDA_KERNEL_IMAGE_FIELDS(add_rms_norm), &addRmsMarshal},
    {"silu_mul_q8x4", PCUDA_KERNEL_IMAGE_FIELDS(silu_mul_q8x4), &siluQMarshal, &siluQExtent, &siluQChunk},
    {"add_sumsq", PCUDA_KERNEL_IMAGE_FIELDS(add_sumsq), &sumsqMarshal, &sumsqExtent, &sumsqChunk},
    {"norm_q8x4", PCUDA_KERNEL_IMAGE_FIELDS(norm_q8x4), &normQMarshal, &normQExtent, &normQChunk},
    {"rms_norm", PCUDA_KERNEL_IMAGE_FIELDS(rms_norm), &rmsMarshal, &rmsExtent, &rmsChunk},
    {"rope", PCUDA_KERNEL_IMAGE_FIELDS(rope), &ropeMarshal},
    {"kv_append", PCUDA_KERNEL_IMAGE_FIELDS(kv_append), &kvMarshal},
    {"attn_scores", PCUDA_KERNEL_IMAGE_FIELDS(attn_scores), &scoresMarshal, &scoresExtent, &scoresChunk},
    {"softmax", PCUDA_KERNEL_IMAGE_FIELDS(softmax), &softmaxMarshal, &softmaxExtent, &softmaxChunk},
    {"attn_values", PCUDA_KERNEL_IMAGE_FIELDS(attn_values), &valuesMarshal, &valuesExtent, &valuesChunk},
    {"silu_mul", PCUDA_KERNEL_IMAGE_FIELDS(silu_mul), &siluMarshal, &siluExtent, &siluChunk},
};

} // namespace

pcuda::Error registerKernels() noexcept {
    for (const auto& e : kEntries)
        if (pcuda::Error r = pcudaRegisterKernel(&e); r != pcuda::Error::Success) return r;
    return pcuda::Error::Success;
}

} // namespace llm
