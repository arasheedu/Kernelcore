#pragma once

// Kernel registration contract between typed adapters and the runtime.
// Booth --cpu kernels use the System V AMD64 convention: up to 6 integer/pointer
// arguments in registers (the last one is Booth's hidden nthreads) and up to 8
// float arguments in xmm registers. Every integer goes in as a full 64-bit value.

#include <pcuda/error.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace pcuda {

struct LaunchArgs {
    uint64_t gpr[6] = {};   // rdi, rsi, rdx, rcx, r8, r9
    unsigned n_gpr = 0;
    uint64_t xmm[8] = {};   // xmm0..xmm7 low 64 bits
    unsigned n_xmm = 0;
    int64_t  nthreads = 0;  // Booth hidden parameter, also appended to gpr by ArgPacker::finish
};

// Passed to adapters so they can check that device pointers are valid runtime allocations.
class LaunchContext {
public:
    virtual ~LaunchContext() = default;
    // Success if [ptr, ptr + bytes) lies inside one live allocation. bytes == 0 always succeeds.
    virtual Error checkDeviceRange(const void* ptr, size_t bytes, const char* what) const noexcept = 0;
};

struct KernelEntry {
    const char*          name = nullptr;          // exact symbol name in the Booth object
    const unsigned char* image = nullptr;         // embedded ELF bytes
    size_t               image_size = 0;
    const char*          image_sha256 = nullptr;  // lowercase hex, checked before loading
    const char*          provenance = nullptr;    // how the object was produced (exe/wsl/prebuilt)

    // Validate typed arguments and fill LaunchArgs (including nthreads).
    Error (*marshal)(const void* typed_args, const LaunchContext& ctx, LaunchArgs* out) noexcept = nullptr;

    // Optional: chunked execution for independent 1-D kernels. Both must be set to opt in.
    // extent: total number of logical threads. marshal_chunk: args for [begin, begin + length).
    Error (*extent)(const void* typed_args, int64_t* total) noexcept = nullptr;
    Error (*marshal_chunk)(const void* typed_args, int64_t begin, int64_t length, LaunchArgs* out) noexcept = nullptr;
};

// Builds LaunchArgs in declaration order. Integers are sign-extended to 64 bits.
class ArgPacker {
public:
    explicit ArgPacker(LaunchArgs& out) noexcept : out_(out) { out_ = LaunchArgs{}; }

    ArgPacker& ptr(const void* p) noexcept { return gpr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))); }
    ArgPacker& i32(int32_t v) noexcept { return gpr(static_cast<uint64_t>(static_cast<int64_t>(v))); }
    ArgPacker& i64(int64_t v) noexcept { return gpr(static_cast<uint64_t>(v)); }
    ArgPacker& f32(float v) noexcept {
        if (out_.n_xmm >= 8) { overflow_ = true; return *this; }
        out_.xmm[out_.n_xmm++] = std::bit_cast<uint32_t>(v);
        return *this;
    }

    // Appends Booth's hidden nthreads argument. Fails if the kernel has too many arguments.
    Error finish(int64_t nthreads) noexcept {
        if (nthreads < 0) return Error::InvalidValue;
        gpr(static_cast<uint64_t>(nthreads));
        if (overflow_) return Error::UnsupportedOperation;
        out_.nthreads = nthreads;
        return Error::Success;
    }

private:
    ArgPacker& gpr(uint64_t v) noexcept {
        if (out_.n_gpr >= 6) { overflow_ = true; return *this; }
        out_.gpr[out_.n_gpr++] = v;
        return *this;
    }
    LaunchArgs& out_;
    bool overflow_ = false;
};

} // namespace pcuda

// Declares the symbols produced by booth_add_kernel() for an embedded kernel object.
#define PCUDA_DECLARE_KERNEL_IMAGE(kname)                                   \
    extern "C" const unsigned char pcuda_kimg_##kname[];                   \
    extern "C" const std::size_t   pcuda_kimg_##kname##_size;              \
    extern "C" const char          pcuda_kimg_##kname##_sha256[];          \
    extern "C" const char          pcuda_kimg_##kname##_provenance[]

#define PCUDA_KERNEL_IMAGE_FIELDS(kname)                                    \
    pcuda_kimg_##kname, pcuda_kimg_##kname##_size,                          \
    pcuda_kimg_##kname##_sha256, pcuda_kimg_##kname##_provenance
