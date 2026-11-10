#pragma once

// Contains hardware exceptions raised inside Booth kernel code.
//
// A vectored exception handler acts only while the current thread is inside a
// KernelScope. It records FaultInfo and resumes at kc_sysv_recover with rsp set
// to the thunk's saved anchor, so the thunk restores every non-volatile register and
// returns 1. No heap allocation, locks or I/O happen in the handler.

#include <kernelcore/error.hpp>
#include <kernelcore/kernel_registry.hpp>

#include <cstddef>

namespace kc::cpu {

// Reference-counted installation of the vectored handler.
void acquireFaultGuard();
void releaseFaultGuard() noexcept;

class KernelScope {
public:
    KernelScope(const char* kernel_name, const void* image_base, size_t image_size) noexcept;
    ~KernelScope() noexcept;
    KernelScope(const KernelScope&) = delete;
    KernelScope& operator=(const KernelScope&) = delete;

    void**           anchorSlot() noexcept;
    bool             faulted() const noexcept;
    const FaultInfo& fault() const noexcept;
};

struct InvokeResult {
    bool      faulted = false;
    FaultInfo fault{};
};

// Calls a System V entry point with fault containment on the current thread.
// After a stack overflow the guard page is re-armed (_resetstkoflw); if that fails
// the process is terminated with __fastfail because the thread can no longer be trusted.
InvokeResult invokeSysV(const void* entry, const LaunchArgs& args, const char* kernel_name,
                        const void* image_base, size_t image_size) noexcept;

} // namespace kc::cpu

extern "C" int kc_sysv_invoke(const void* entry, const uint64_t* gpr6, const uint64_t* xmm8,
                                 void** anchor_slot);
extern "C" void kc_sysv_recover();
extern "C" void kc_sysv_native_bridge();  // src/cpu/sysv_shims.asm
