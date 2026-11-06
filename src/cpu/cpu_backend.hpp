#pragma once

#include "../allocation_registry.hpp"
#include "booth_image.hpp"

#include <pcuda/backend.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace pcuda::cpu {

class CpuBackend final : public IBackend {
public:
    CpuBackend(AllocationRegistry& registry, bool guard_pages, bool large_pages);
    ~CpuBackend() override;

    const char* name() const noexcept override { return "cpu-booth-x86_64"; }
    Error allocate(void** ptr, size_t bytes) noexcept override;
    Error release(void* ptr) noexcept override;
    Error copy(void* dst, const void* src, size_t bytes, CopyKind kind) noexcept override;
    Error launch(const KernelEntry& entry, const LaunchArgs& args) noexcept override;

    // Loads (once) and returns the mapped image for a kernel. Load failures are cached.
    Error image(const KernelEntry& entry, const BoothImage** out);
    void  unloadAll() noexcept;

    const FaultInfo& lastFault() const noexcept { return last_fault_; }

    // Code generation path (see pcuda::Codegen). resolve() applies Auto and per-kernel fallbacks.
    void    setCodegen(Codegen c) noexcept { codegen_.store(c); }
    Codegen codegen() const noexcept { return codegen_.load(); }
    Codegen resolve(const KernelEntry& entry) const noexcept;
    static bool cpuHasAvx2() noexcept;

private:
    struct Slot {
        std::unique_ptr<BoothImage> image;
        Error                       error = Error::Success;
        std::string                 detail;
    };
    AllocationRegistry&         registry_;
    bool                        guard_pages_;
    bool                        large_pages_;  // blocks of >= 2 MB use large pages
    std::mutex                  images_mutex_;
    std::map<std::string, Slot> images_;
    std::mutex                  fault_mutex_;
    FaultInfo                   last_fault_{};
    std::atomic<Codegen>        codegen_{Codegen::Auto};
};

// Copies with SEH protection so an unreadable/unwritable host pointer becomes an error.
bool protectedMove(void* dst, const void* src, size_t bytes) noexcept;
bool protectedFill(void* dst, int value, size_t bytes) noexcept;

} // namespace pcuda::cpu
