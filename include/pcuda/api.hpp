#pragma once

// pcuda: a CUDA-inspired runtime that runs Booth-compiled kernels on Windows x64 CPUs.
// Not CUDA-compatible: names are prefixed and semantics are documented per call.
// Every function is noexcept and reports failure through pcuda::Error.

#include <pcuda/error.hpp>
#include <pcuda/kernel_registry.hpp>
#include <pcuda/types.hpp>

#include <cstddef>
#include <cstdint>

// ---- lifecycle -------------------------------------------------------------
pcuda::Error pcudaInit() noexcept;       // idempotent
pcuda::Error pcudaShutdown() noexcept;   // idempotent; drains streams, frees everything
pcuda::Error pcudaDeviceReset() noexcept; // frees everything, clears ContextPoisoned

// ---- device ----------------------------------------------------------------
pcuda::Error pcudaGetDeviceCount(int* count) noexcept;
pcuda::Error pcudaGetDeviceProperties(pcuda::DeviceProperties* props, int device) noexcept;
pcuda::Error pcudaSetDevice(int device) noexcept;
pcuda::Error pcudaGetDevice(int* device) noexcept;
pcuda::Error pcudaDeviceSynchronize() noexcept; // waits for all streams; returns first pending error

// ---- memory ----------------------------------------------------------------
pcuda::Error pcudaMalloc(void** ptr, size_t bytes) noexcept;
pcuda::Error pcudaFree(void* ptr) noexcept;  // waits for all streams first
pcuda::Error pcudaMemcpy(void* dst, const void* src, size_t bytes, pcuda::CopyKind kind) noexcept;
pcuda::Error pcudaMemset(void* dst, int value, size_t bytes) noexcept;

// ---- kernels ---------------------------------------------------------------
pcuda::Error pcudaRegisterKernel(const pcuda::KernelEntry* entry) noexcept; // entry must outlive the runtime
// stream == nullptr: synchronous, returns the kernel's result. Otherwise async; errors surface at sync.
pcuda::Error pcudaLaunchKernel(const char* name, const void* typed_args, pcudaStream_t stream = nullptr) noexcept;
// Splits an opt-in 1-D kernel into disjoint chunks run on a worker pool. Host-side decomposition only.
pcuda::Error pcudaLaunchKernelChunked(const char* name, const void* typed_args, unsigned workers,
                                      int64_t min_chunk = 4096, pcudaStream_t stream = nullptr) noexcept;

// ---- streams and events ----------------------------------------------------
pcuda::Error pcudaStreamCreate(pcudaStream_t* stream) noexcept;
pcuda::Error pcudaStreamDestroy(pcudaStream_t stream) noexcept;    // drains first
pcuda::Error pcudaStreamSynchronize(pcudaStream_t stream) noexcept;
pcuda::Error pcudaStreamQuery(pcudaStream_t stream) noexcept;      // Success or NotReady
pcuda::Error pcudaLaunchHostFunc(pcudaStream_t stream, void (*fn)(void*), void* user) noexcept;
pcuda::Error pcudaEventCreate(pcudaEvent_t* event) noexcept;
pcuda::Error pcudaEventDestroy(pcudaEvent_t event) noexcept;
pcuda::Error pcudaEventRecord(pcudaEvent_t event, pcudaStream_t stream = nullptr) noexcept;
pcuda::Error pcudaEventSynchronize(pcudaEvent_t event) noexcept;
pcuda::Error pcudaEventQuery(pcudaEvent_t event) noexcept;         // Success or NotReady
pcuda::Error pcudaStreamWaitEvent(pcudaStream_t stream, pcudaEvent_t event) noexcept; // UnsupportedOperation in POC

// ---- diagnostics -----------------------------------------------------------
pcuda::Error pcudaGetLastFault(pcuda::FaultInfo* info) noexcept;
const char*  pcudaGetErrorString(pcuda::Error error) noexcept;
// Thread-local diagnostic text from the most recent pcuda call on this thread. Every call
// resets it, so read it immediately after the call that failed. Empty when there is nothing to say.
const char*  pcudaGetLastErrorDetail() noexcept;
