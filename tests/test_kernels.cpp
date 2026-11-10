#include "test_kernels.hpp"

#include <array>
#include <limits>

KC_DECLARE_KERNEL_IMAGE(vector_add);
KC_DECLARE_KERNEL_IMAGE(oob_write);
KC_DECLARE_KERNEL_IMAGE(int_div);
KC_DECLARE_KERNEL_IMAGE(deep_recursion);
KC_DECLARE_KERNEL_IMAGE(scale_add);
KC_DECLARE_KERNEL_IMAGE(trig_mix);

namespace testkernels {
namespace {

using kc::ArgPacker;
using kc::Error;
using kc::LaunchArgs;
using kc::LaunchContext;

Error oobWrite(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const OobWriteArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    // Validates only [0, count): the kernel's overrun is the point of the test.
    if (Error e = ctx.checkDeviceRange(a.out, size_t(a.count) * sizeof(float), "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.out).i32(a.count);
    return pk.finish(a.count);
}

Error intDiv(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const IntDivArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    const size_t bytes = size_t(a.count) * sizeof(int);
    if (Error e = ctx.checkDeviceRange(a.a, bytes, "a"); e != Error::Success) return e;
    if (Error e = ctx.checkDeviceRange(a.out, bytes, "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.a).ptr(a.out).i32(a.divisor).i32(a.count);
    return pk.finish(a.count);
}

Error deepRecursion(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const DeepRecursionArgs*>(p);
    if (a.count < 0 || a.depth < 0) return Error::InvalidValue;
    if (Error e = ctx.checkDeviceRange(a.out, size_t(a.count) * sizeof(int), "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.out).i32(a.depth).i32(a.count);
    return pk.finish(a.count);
}

Error scaleAdd(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const ScaleAddArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    const size_t bytes = size_t(a.count) * sizeof(float);
    if (Error e = ctx.checkDeviceRange(a.x, bytes, "x"); e != Error::Success) return e;
    if (Error e = ctx.checkDeviceRange(a.out, bytes, "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.out).f32(a.scale).f32(a.bias).i32(a.count);
    return pk.finish(a.count);
}

Error trigMix(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const TrigMixArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    const size_t bytes = size_t(a.count) * sizeof(float);
    if (Error e = ctx.checkDeviceRange(a.x, bytes, "x"); e != Error::Success) return e;
    if (Error e = ctx.checkDeviceRange(a.out, bytes, "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.x).ptr(a.out).i32(a.count);
    return pk.finish(a.count);
}

// Chunk hooks (same contract as vector_add): offset the pointers to the chunk start and
// pass count = nthreads = chunk length. Valid because each element is independent.
bool chunkOk(int count, int64_t begin, int64_t length) noexcept {
    return begin >= 0 && length > 0 && begin + length <= count && length <= std::numeric_limits<int32_t>::max();
}

Error scaleAddExtent(const void* p, int64_t* total) noexcept {
    const auto& a = *static_cast<const ScaleAddArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    *total = a.count;
    return Error::Success;
}

Error scaleAddChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const ScaleAddArgs*>(p);
    if (!chunkOk(a.count, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x + begin).ptr(a.out + begin).f32(a.scale).f32(a.bias).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

Error trigMixExtent(const void* p, int64_t* total) noexcept {
    const auto& a = *static_cast<const TrigMixArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    *total = a.count;
    return Error::Success;
}

Error trigMixChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const TrigMixArgs*>(p);
    if (!chunkOk(a.count, begin, length)) return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.x + begin).ptr(a.out + begin).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

const kc::KernelEntry kEntries[] = {
    {"oob_write", KC_KERNEL_IMAGE_FIELDS(oob_write), &oobWrite},
    {"int_div", KC_KERNEL_IMAGE_FIELDS(int_div), &intDiv},
    {"deep_recursion", KC_KERNEL_IMAGE_FIELDS(deep_recursion), &deepRecursion},
    {"scale_add", KC_KERNEL_IMAGE_FIELDS(scale_add), &scaleAdd, &scaleAddExtent, &scaleAddChunk},
    {"trig_mix", KC_KERNEL_IMAGE_FIELDS(trig_mix), &trigMix, &trigMixExtent, &trigMixChunk},
};

} // namespace

Error registerAll() {
    for (const auto& e : kEntries)
        if (Error r = kcRegisterKernel(&e); r != Error::Success) return r;
    return Error::Success;
}

std::span<const ImageRef> images() {
    static const std::array<ImageRef, 6> k{{
        {"vector_add", kc_kimg_vector_add, kc_kimg_vector_add_size, kc_kimg_vector_add_sha256},
        {"oob_write", kc_kimg_oob_write, kc_kimg_oob_write_size, kc_kimg_oob_write_sha256},
        {"int_div", kc_kimg_int_div, kc_kimg_int_div_size, kc_kimg_int_div_sha256},
        {"deep_recursion", kc_kimg_deep_recursion, kc_kimg_deep_recursion_size,
         kc_kimg_deep_recursion_sha256},
        {"scale_add", kc_kimg_scale_add, kc_kimg_scale_add_size, kc_kimg_scale_add_sha256},
        {"trig_mix", kc_kimg_trig_mix, kc_kimg_trig_mix_size, kc_kimg_trig_mix_sha256},
    }};
    return k;
}

} // namespace testkernels
