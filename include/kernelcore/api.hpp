#pragma once

// Kernelcore: a CUDA-inspired runtime that runs Booth-compiled kernels on Windows x64 CPUs.
// Not CUDA-compatible: names are prefixed and semantics are documented per call.
// Every function is noexcept and reports failure through kc::Error.

#include <kernelcore/error.hpp>
#include <kernelcore/kernel_registry.hpp>
#include <kernelcore/types.hpp>

#include <cstddef>
#include <cstdint>

// ---- lifecycle -------------------------------------------------------------
kc::Error kcInit() noexcept;       // idempotent
kc::Error kcShutdown() noexcept;   // idempotent; drains streams, frees everything
kc::Error kcDeviceReset() noexcept; // frees everything, clears ContextPoisoned

// ---- device ----------------------------------------------------------------
kc::Error kcGetDeviceCount(int* count) noexcept;
kc::Error kcGetDeviceProperties(kc::DeviceProperties* props, int device) noexcept;
kc::Error kcSetDevice(int device) noexcept;
kc::Error kcGetDevice(int* device) noexcept;
kc::Error kcDeviceSynchronize() noexcept; // waits for all streams; returns first pending error

// ---- memory ----------------------------------------------------------------
kc::Error kcMalloc(void** ptr, size_t bytes) noexcept;
kc::Error kcFree(void* ptr) noexcept;  // waits for all streams first
kc::Error kcMemcpy(void* dst, const void* src, size_t bytes, kc::CopyKind kind) noexcept;
kc::Error kcMemset(void* dst, int value, size_t bytes) noexcept;

// ---- kernels ---------------------------------------------------------------
kc::Error kcRegisterKernel(const kc::KernelEntry* entry) noexcept; // entry must outlive the runtime
// stream == nullptr: synchronous, returns the kernel's result. Otherwise async; errors surface at sync.
kc::Error kcLaunchKernel(const char* name, const void* typed_args, kcStream_t stream = nullptr) noexcept;
// Splits an opt-in 1-D kernel into disjoint chunks run on a worker pool. Host-side decomposition only.
// At most `workers` threads (the caller included) run the chunks, also in a recorded graph.
kc::Error kcLaunchKernelChunked(const char* name, const void* typed_args, unsigned workers,
                                      int64_t min_chunk = 4096, kcStream_t stream = nullptr) noexcept;

// ---- streams and events ----------------------------------------------------
kc::Error kcStreamCreate(kcStream_t* stream) noexcept;
kc::Error kcStreamDestroy(kcStream_t stream) noexcept;    // drains first
kc::Error kcStreamSynchronize(kcStream_t stream) noexcept;
kc::Error kcStreamQuery(kcStream_t stream) noexcept;      // Success or NotReady
kc::Error kcLaunchHostFunc(kcStream_t stream, void (*fn)(void*), void* user) noexcept;
kc::Error kcEventCreate(kcEvent_t* event) noexcept;
kc::Error kcEventDestroy(kcEvent_t event) noexcept;
kc::Error kcEventRecord(kcEvent_t event, kcStream_t stream = nullptr) noexcept;
kc::Error kcEventSynchronize(kcEvent_t event) noexcept;
kc::Error kcEventQuery(kcEvent_t event) noexcept;         // Success or NotReady
kc::Error kcStreamWaitEvent(kcStream_t stream, kcEvent_t event) noexcept; // UnsupportedOperation in POC

// ---- graphs (like CUDA graphs: record launches once, run them as one job) ----
// Between BeginCapture and EndCapture, kernel launches on the stream (from any thread) are
// validated and recorded instead of run, and return Success. Copies, memsets, frees and device
// synchronization are refused (UnsupportedOperation) while any stream captures, as are
// synchronizing, host functions and event records on the capturing stream itself.
kc::Error kcStreamBeginCapture(kcStream_t stream) noexcept;
kc::Error kcStreamEndCapture(kcStream_t stream, kcGraph_t* graph) noexcept;
// Runs the recorded kernels in order as one pool job: each kernel's chunks run concurrently and
// the next kernel starts when they are all done, without a stream hand-off or thread wake-up
// per kernel. Results match running the launches one by one. After a failing kernel, later
// kernels are skipped; the error names the graph node. stream == nullptr: synchronous.
kc::Error kcGraphLaunch(kcGraph_t graph, kcStream_t stream = nullptr) noexcept;
// While capturing: the launches recorded between Begin and End are independent of each other and
// run as one step (their chunks are shared out together; the next step waits for all of them).
kc::Error kcStreamConcurrentBegin(kcStream_t stream = nullptr) noexcept;
kc::Error kcStreamConcurrentEnd(kcStream_t stream = nullptr) noexcept;
kc::Error kcGraphDestroy(kcGraph_t graph) noexcept;  // a pending launch keeps its data

// ---- code generation path ---------------------------------------------------
// Process-wide; applies to launches made after the call. Default: env KC_CODEGEN, else Auto.
kc::Error kcSetKernelCodegen(kc::Codegen codegen) noexcept;
kc::Error kcGetKernelCodegen(kc::Codegen* codegen) noexcept;
// The path a launch of `name` would take now (resolves Auto and per-kernel fallbacks).
kc::Error kcGetKernelCodegenFor(const char* name, kc::Codegen* used) noexcept;
const char*  kcCodegenName(kc::Codegen codegen) noexcept;

// ---- diagnostics -----------------------------------------------------------
kc::Error kcGetLastFault(kc::FaultInfo* info) noexcept;
const char*  kcGetErrorString(kc::Error error) noexcept;
// Thread-local diagnostic text from the most recent Kernelcore call on this thread. Every call
// resets it, so read it immediately after the call that failed. Empty when there is nothing to say.
const char*  kcGetLastErrorDetail() noexcept;
