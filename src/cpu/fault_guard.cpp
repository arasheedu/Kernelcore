#include "fault_guard.hpp"

#include <windows.h>

#include <malloc.h>   // _resetstkoflw
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace pcuda::cpu {
namespace {

struct ThreadKernelState {
    bool        in_kernel = false;
    bool        faulted = false;
    void*       anchor = nullptr;
    const char* name = nullptr;
    uintptr_t   base = 0;
    size_t      size = 0;
    FaultInfo   fault{};
};

thread_local ThreadKernelState t_state;

std::mutex g_mutex;
int        g_refs = 0;
PVOID      g_handler = nullptr;

bool isRecoverable(DWORD code) noexcept {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_IN_PAGE_ERROR:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_INT_OVERFLOW:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_BREAKPOINT:          // int3 padding around kernel code
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_FLT_DENORMAL_OPERAND:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INEXACT_RESULT:
    case EXCEPTION_FLT_INVALID_OPERATION:
    case EXCEPTION_FLT_OVERFLOW:
    case EXCEPTION_FLT_STACK_CHECK:
    case EXCEPTION_FLT_UNDERFLOW:
    case 0xC00002B4:                    // STATUS_FLOAT_MULTIPLE_FAULTS
    case 0xC00002B5:                    // STATUS_FLOAT_MULTIPLE_TRAPS
        return true;
    default:
        return false;
    }
}

LONG CALLBACK vehHandler(PEXCEPTION_POINTERS ep) {
    ThreadKernelState& s = t_state;
    if (!s.in_kernel || s.anchor == nullptr) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD* er = ep->ExceptionRecord;
    if (!isRecoverable(er->ExceptionCode)) return EXCEPTION_CONTINUE_SEARCH;

    FaultInfo& f = s.fault;
    f.code = er->ExceptionCode;
    f.instruction = ep->ContextRecord->Rip;
    if ((er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION || er->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        er->NumberParameters >= 2) {
        f.access_type = static_cast<uint32_t>(er->ExceptionInformation[0]);
        f.fault_address = er->ExceptionInformation[1];
    }
    const uintptr_t rip = static_cast<uintptr_t>(ep->ContextRecord->Rip);
    f.in_kernel_image = s.base != 0 && rip >= s.base && rip - s.base < s.size;
    f.code_offset = f.in_kernel_image ? rip - s.base : 0;
    size_t i = 0;
    if (s.name)
        for (; i + 1 < sizeof f.kernel && s.name[i]; ++i) f.kernel[i] = s.name[i];
    f.kernel[i] = '\0';

    s.faulted = true;
    s.in_kernel = false;
    ep->ContextRecord->Rsp = reinterpret_cast<DWORD64>(s.anchor);
    ep->ContextRecord->Rip = reinterpret_cast<DWORD64>(&pcuda_sysv_recover);
    s.anchor = nullptr;
    return EXCEPTION_CONTINUE_EXECUTION;
}

} // namespace

void acquireFaultGuard() {
    std::lock_guard lk(g_mutex);
    if (g_refs++ == 0) {
        g_handler = AddVectoredExceptionHandler(1, &vehHandler);
        if (!g_handler) {
            g_refs = 0;
            throw std::runtime_error("AddVectoredExceptionHandler failed");
        }
    }
}

void releaseFaultGuard() noexcept {
    std::lock_guard lk(g_mutex);
    if (g_refs > 0 && --g_refs == 0 && g_handler) {
        RemoveVectoredExceptionHandler(g_handler);
        g_handler = nullptr;
    }
}

KernelScope::KernelScope(const char* kernel_name, const void* image_base, size_t image_size) noexcept {
    ThreadKernelState& s = t_state;
    if (s.in_kernel) __fastfail(FAST_FAIL_INVALID_ARG);  // nested kernel calls are a runtime bug
    s = ThreadKernelState{};
    s.name = kernel_name;
    s.base = reinterpret_cast<uintptr_t>(image_base);
    s.size = image_size;
    s.in_kernel = true;
}

KernelScope::~KernelScope() noexcept {
    t_state.in_kernel = false;
    t_state.anchor = nullptr;
}

void** KernelScope::anchorSlot() noexcept { return &t_state.anchor; }
bool KernelScope::faulted() const noexcept { return t_state.faulted; }
const FaultInfo& KernelScope::fault() const noexcept { return t_state.fault; }

InvokeResult invokeSysV(const void* entry, const LaunchArgs& args, const char* kernel_name,
                        const void* image_base, size_t image_size) noexcept {
    InvokeResult r;
    {
        KernelScope scope(kernel_name, image_base, image_size);
        const int rc = pcuda_sysv_invoke(entry, args.gpr, args.xmm, scope.anchorSlot());
        if (rc != 0 || scope.faulted()) {
            r.faulted = true;
            r.fault = scope.fault();
        }
    }
    if (r.faulted && r.fault.code == EXCEPTION_STACK_OVERFLOW) {
        if (!_resetstkoflw()) __fastfail(FAST_FAIL_FATAL_APP_EXIT);
    }
    return r;
}

} // namespace pcuda::cpu
