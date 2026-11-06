// Typed adapter for kernels/vector_add.cu.
// Booth 0.6.1 symbol: vector_add(const float* a, const float* b, float* out, int count, <nthreads>)
// System V registers: rdi=a, rsi=b, rdx=out, rcx=count (sign-extended), r8=nthreads (64-bit).

#include <pcuda/kernels.hpp>

#include <limits>

PCUDA_DECLARE_KERNEL_IMAGE(vector_add);

namespace pcuda::kernels {
namespace {

Error marshal(const void* p, const LaunchContext& ctx, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const VectorAddArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    const size_t bytes = static_cast<size_t>(a.count) * sizeof(float);
    if (Error e = ctx.checkDeviceRange(a.a, bytes, "a"); e != Error::Success) return e;
    if (Error e = ctx.checkDeviceRange(a.b, bytes, "b"); e != Error::Success) return e;
    if (Error e = ctx.checkDeviceRange(a.out, bytes, "out"); e != Error::Success) return e;
    ArgPacker pk(*out);
    pk.ptr(a.a).ptr(a.b).ptr(a.out).i32(a.count);
    return pk.finish(a.count);
}

Error extent(const void* p, int64_t* total) noexcept {
    const auto& a = *static_cast<const VectorAddArgs*>(p);
    if (a.count < 0) return Error::InvalidValue;
    *total = a.count;
    return Error::Success;
}

// Chunk contract (Booth: threadIdx.x in [0, nthreads), blockIdx.x == 0): offset every
// pointer to the chunk start and pass count = nthreads = chunk length.
Error marshalChunk(const void* p, int64_t begin, int64_t length, LaunchArgs* out) noexcept {
    const auto& a = *static_cast<const VectorAddArgs*>(p);
    if (begin < 0 || length <= 0 || begin + length > a.count || length > std::numeric_limits<int32_t>::max())
        return Error::InvalidValue;
    ArgPacker pk(*out);
    pk.ptr(a.a + begin).ptr(a.b + begin).ptr(a.out + begin).i32(static_cast<int32_t>(length));
    return pk.finish(length);
}

} // namespace

const KernelEntry& vectorAddEntry() noexcept {
    static const KernelEntry e{"vector_add", PCUDA_KERNEL_IMAGE_FIELDS(vector_add), &marshal, &extent, &marshalChunk};
    return e;
}

} // namespace pcuda::kernels
