#include "test_kernels.hpp"

#include <array>

PCUDA_DECLARE_KERNEL_IMAGE(vector_add);
PCUDA_DECLARE_KERNEL_IMAGE(oob_write);
PCUDA_DECLARE_KERNEL_IMAGE(int_div);
PCUDA_DECLARE_KERNEL_IMAGE(deep_recursion);
PCUDA_DECLARE_KERNEL_IMAGE(scale_add);
PCUDA_DECLARE_KERNEL_IMAGE(trig_mix);

namespace testkernels {
namespace {

using pcuda::ArgPacker;
using pcuda::Error;
using pcuda::LaunchArgs;
using pcuda::LaunchContext;

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

const pcuda::KernelEntry kEntries[] = {
    {"oob_write", PCUDA_KERNEL_IMAGE_FIELDS(oob_write), &oobWrite},
    {"int_div", PCUDA_KERNEL_IMAGE_FIELDS(int_div), &intDiv},
    {"deep_recursion", PCUDA_KERNEL_IMAGE_FIELDS(deep_recursion), &deepRecursion},
    {"scale_add", PCUDA_KERNEL_IMAGE_FIELDS(scale_add), &scaleAdd},
    {"trig_mix", PCUDA_KERNEL_IMAGE_FIELDS(trig_mix), &trigMix},
};

} // namespace

Error registerAll() {
    for (const auto& e : kEntries)
        if (Error r = pcudaRegisterKernel(&e); r != Error::Success) return r;
    return Error::Success;
}

std::span<const ImageRef> images() {
    static const std::array<ImageRef, 6> k{{
        {"vector_add", pcuda_kimg_vector_add, pcuda_kimg_vector_add_size, pcuda_kimg_vector_add_sha256},
        {"oob_write", pcuda_kimg_oob_write, pcuda_kimg_oob_write_size, pcuda_kimg_oob_write_sha256},
        {"int_div", pcuda_kimg_int_div, pcuda_kimg_int_div_size, pcuda_kimg_int_div_sha256},
        {"deep_recursion", pcuda_kimg_deep_recursion, pcuda_kimg_deep_recursion_size,
         pcuda_kimg_deep_recursion_sha256},
        {"scale_add", pcuda_kimg_scale_add, pcuda_kimg_scale_add_size, pcuda_kimg_scale_add_sha256},
        {"trig_mix", pcuda_kimg_trig_mix, pcuda_kimg_trig_mix_size, pcuda_kimg_trig_mix_sha256},
    }};
    return k;
}

} // namespace testkernels
