#pragma once

#include <cstdint>

namespace kc {

// Stable error codes. Values are part of the POC contract; append only.
enum class Error : int {
    Success = 0,
    NotInitialized,         // API used before kcInit or after kcShutdown
    InvalidValue,           // bad argument (null, negative count, bad enum, unreadable host pointer)
    InvalidDevice,          // device index other than 0
    OutOfMemory,
    InvalidPointer,         // not a live runtime allocation, or range outside it
    InvalidResourceHandle,  // unknown stream/event handle
    UnsupportedOperation,   // feature deliberately not implemented in the POC
    InvalidKernelImage,     // embedded Booth object failed validation
    KernelNotFound,         // no kernel registered under that name
    DynamicCodeBlocked,     // process policy (ACG) forbids executable memory
    LaunchFailure,          // launch rejected before execution
    KernelFault,            // hardware exception inside kernel code (sticky)
    ContextPoisoned,        // returned by most calls after a KernelFault until kcDeviceReset
    NotReady,               // query: work still pending
    InternalError           // unexpected internal failure (bug); see kcGetLastErrorDetail
};

// Filled when a kernel faults. Read with kcGetLastFault.
struct FaultInfo {
    uint32_t code = 0;   // e.g. 0xC0000005 access violation
    uint64_t fault_address = 0;    // data address for access violations
    uint32_t access_type = 0;      // 0 = read, 1 = write, 8 = execute (access violations only)
    uint64_t instruction = 0;      // RIP at the fault
    bool     in_kernel_image = false;
    uint64_t code_offset = 0;      // RIP - image base, valid when in_kernel_image
    char     kernel[64] = {};
};

} // namespace kc
