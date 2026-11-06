#include "cpu_backend.hpp"

#include "../detail.hpp"
#include "fault_guard.hpp"

#include <windows.h>

#include <intrin.h>
#include <immintrin.h>

#include <cstring>
#include <format>

namespace pcuda::cpu {

// Kept free of C++ objects with destructors so __try is allowed.
bool protectedMove(void* dst, const void* src, size_t bytes) noexcept {
    __try {
        std::memmove(dst, src, bytes);
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION || GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
                    ? EXCEPTION_EXECUTE_HANDLER
                    : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

bool protectedFill(void* dst, int value, size_t bytes) noexcept {
    __try {
        std::memset(dst, value, bytes);
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION || GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
                    ? EXCEPTION_EXECUTE_HANDLER
                    : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

CpuBackend::CpuBackend(AllocationRegistry& registry, bool guard_pages)
    : registry_(registry), guard_pages_(guard_pages) {
    acquireFaultGuard();
}

CpuBackend::~CpuBackend() {
    unloadAll();
    releaseFaultGuard();
}

Error CpuBackend::allocate(void** ptr, size_t bytes) noexcept {
    HostBlock b;
    const Error e = allocateBlock(bytes, guard_pages_, b);
    if (e != Error::Success) {
        detail::setDetail(std::format("allocation of {} bytes failed", bytes));
        return e;
    }
    try {
        const Error r = registry_.insert(b, 0);
        if (r != Error::Success) {
            releaseBlock(b);
            return r;
        }
    } catch (...) {
        releaseBlock(b);
        return Error::OutOfMemory;
    }
    *ptr = b.user;
    return Error::Success;
}

Error CpuBackend::release(void* ptr) noexcept {
    try {
        HostBlock b;
        std::string why;
        const Error e = registry_.remove(ptr, b, why);
        if (e != Error::Success) {
            detail::setDetail(why);
            return e;
        }
        releaseBlock(b);
        return Error::Success;
    } catch (...) {
        return Error::InternalError;
    }
}

Error CpuBackend::copy(void* dst, const void* src, size_t bytes, CopyKind kind) noexcept {
    try {
        std::string why;
        const bool dst_dev = kind == CopyKind::HostToDevice || kind == CopyKind::DeviceToDevice;
        const bool src_dev = kind == CopyKind::DeviceToHost || kind == CopyKind::DeviceToDevice;
        if (dst_dev) {
            if (Error e = registry_.checkRange(dst, bytes, why); e != Error::Success) {
                detail::setDetail("destination: " + why);
                return e;
            }
        }
        if (src_dev) {
            if (Error e = registry_.checkRange(src, bytes, why); e != Error::Success) {
                detail::setDetail("source: " + why);
                return e;
            }
        }
        if (!protectedMove(dst, src, bytes)) {
            detail::setDetail("host pointer is not accessible for the requested range");
            return Error::InvalidValue;
        }
        return Error::Success;
    } catch (...) {
        return Error::InternalError;
    }
}

Error CpuBackend::image(const KernelEntry& entry, const BoothImage** out) {
    std::lock_guard lk(images_mutex_);
    auto [it, inserted] = images_.try_emplace(entry.name);
    Slot& s = it->second;
    if (inserted) {
        s.error = BoothImage::load({entry.image, entry.image_size}, entry.name,
                                   entry.image_sha256 ? entry.image_sha256 : "", s.image, s.detail);
    }
    if (s.error != Error::Success) {
        detail::setDetail(std::format("kernel '{}': {}", entry.name, s.detail));
        return s.error;
    }
    *out = s.image.get();
    return Error::Success;
}

void CpuBackend::unloadAll() noexcept {
    std::lock_guard lk(images_mutex_);
    images_.clear();
}

bool CpuBackend::cpuHasAvx2() noexcept {
    static const bool has = [] {
        int r[4] = {};
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const bool osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
        if (!osxsave || !avx) return false;
        if ((_xgetbv(0) & 0x6) != 0x6) return false;  // OS saves XMM and YMM state
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;  // AVX2
    }();
    return has;
}

Codegen CpuBackend::resolve(const KernelEntry& entry) const noexcept {
    const Codegen want = codegen_.load();
    const NativeKernel* n = entry.native;
    if (want == Codegen::Booth || !n || !n->scalar) return Codegen::Booth;
    if ((want == Codegen::Simd || want == Codegen::Auto) && n->simd && cpuHasAvx2()) return Codegen::Simd;
    return Codegen::Native;
}

Error CpuBackend::launch(const KernelEntry& entry, const LaunchArgs& args) noexcept {
    try {
        if (args.nthreads == 0) return Error::Success;
        Codegen path = resolve(entry);
        // Fewer threads than one 8-lane group would leave most lanes idle while still paying for
        // masked memory access: the scalar native code is faster there, and bit-identical.
        if (path == Codegen::Simd && args.nthreads < 8) path = Codegen::Native;
        InvokeResult r;
        if (path == Codegen::Booth) {
            const BoothImage* img = nullptr;
            if (Error e = image(entry, &img); e != Error::Success) return e;
            r = invokeSysV(img->entry(), args, entry.name, img->base(), img->size());
        } else {
            // The generated function runs under the same thunk (and fault guard) via a bridge.
            LaunchArgs bridge{};
            bridge.gpr[0] = reinterpret_cast<uint64_t>(&args);
            bridge.gpr[1] = reinterpret_cast<uint64_t>(path == Codegen::Simd ? entry.native->simd : entry.native->scalar);
            bridge.n_gpr = 2;
            r = invokeSysV(reinterpret_cast<const void*>(&pcuda_sysv_native_bridge), bridge, entry.name, nullptr, 0);
        }
        if (r.faulted) {
            {
                std::lock_guard lk(fault_mutex_);
                last_fault_ = r.fault;
            }
            const std::string where = r.fault.in_kernel_image ? std::format(".text+0x{:X}", r.fault.code_offset)
                                      : path == Codegen::Booth ? std::string("outside kernel image")
                                                               : std::format("{} code", path == Codegen::Simd ? "simd" : "native");
            detail::setDetail(std::format("kernel '{}' raised exception 0x{:08X} at {} (address 0x{:X})", entry.name,
                                          r.fault.code, where, r.fault.fault_address));
            return Error::KernelFault;
        }
        return Error::Success;
    } catch (...) {
        return Error::InternalError;
    }
}

} // namespace pcuda::cpu
