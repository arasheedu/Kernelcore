#pragma once

// Backend seam. CpuBackend implements it today; an NVIDIA backend would implement
// the same contract where semantics genuinely match.

#include <kernelcore/error.hpp>
#include <kernelcore/kernel_registry.hpp>
#include <kernelcore/types.hpp>

#include <cstddef>

namespace kc {

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* name() const noexcept = 0;
    virtual Error allocate(void** ptr, size_t bytes) noexcept = 0;
    virtual Error release(void* ptr) noexcept = 0;
    virtual Error copy(void* dst, const void* src, size_t bytes, CopyKind kind) noexcept = 0;
    // Runs one marshalled launch synchronously on the calling thread.
    virtual Error launch(const KernelEntry& entry, const LaunchArgs& args) noexcept = 0;
};

} // namespace kc
