#include "cpu_backend.hpp"

#include "../detail.hpp"
#include "fault_guard.hpp"

#include <windows.h>

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

Error CpuBackend::launch(const KernelEntry& entry, const LaunchArgs& args) noexcept {
    try {
        if (args.nthreads == 0) return Error::Success;
        const BoothImage* img = nullptr;
        if (Error e = image(entry, &img); e != Error::Success) return e;
        const InvokeResult r = invokeSysV(img->entry(), args, entry.name, img->base(), img->size());
        if (r.faulted) {
            {
                std::lock_guard lk(fault_mutex_);
                last_fault_ = r.fault;
            }
            detail::setDetail(std::format("kernel '{}' raised exception 0x{:08X} at {} (address 0x{:X})",
                                          entry.name, r.fault.code,
                                          r.fault.in_kernel_image ? std::format(".text+0x{:X}", r.fault.code_offset)
                                                                  : std::string("outside kernel image"),
                                          r.fault.fault_address));
            return Error::KernelFault;
        }
        return Error::Success;
    } catch (...) {
        return Error::InternalError;
    }
}

} // namespace pcuda::cpu
