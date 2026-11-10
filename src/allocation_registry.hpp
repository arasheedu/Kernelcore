#pragma once

// Tracks every runtime-owned allocation. Ownership is never inferred from pointer values.

#include "cpu/guarded_alloc.hpp"

#include <kernelcore/error.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

namespace kc {

class AllocationRegistry {
public:
    Error insert(const cpu::HostBlock& block, int device);
    // Exact base pointer only. Interior and unknown pointers return InvalidPointer.
    Error remove(const void* base, cpu::HostBlock& out, std::string& detail);
    // [ptr, ptr + bytes) must lie inside one live allocation. Interior pointers allowed.
    Error checkRange(const void* ptr, size_t bytes, std::string& detail) const;
    size_t count() const;
    std::vector<cpu::HostBlock> drain();

private:
    struct Entry {
        cpu::HostBlock block;
        int            device;
    };
    mutable std::shared_mutex     mutex_;
    std::map<uintptr_t, Entry>    map_;
};

} // namespace kc
