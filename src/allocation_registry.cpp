#include "allocation_registry.hpp"

#include <format>
#include <mutex>

namespace kc {

Error AllocationRegistry::insert(const cpu::HostBlock& block, int device) {
    std::unique_lock lk(mutex_);
    const auto key = reinterpret_cast<uintptr_t>(block.user);
    if (!map_.emplace(key, Entry{block, device}).second) return Error::InternalError;
    return Error::Success;
}

Error AllocationRegistry::remove(const void* base, cpu::HostBlock& out, std::string& detail) {
    std::unique_lock lk(mutex_);
    const auto key = reinterpret_cast<uintptr_t>(base);
    auto it = map_.find(key);
    if (it == map_.end()) {
        auto ub = map_.upper_bound(key);
        if (ub != map_.begin()) {
            --ub;
            if (key - ub->first < ub->second.block.user_bytes) {
                detail = std::format("pointer {} is interior to allocation {} (offset {})", base,
                                     static_cast<const void*>(ub->second.block.user), key - ub->first);
                return Error::InvalidPointer;
            }
        }
        detail = std::format("pointer {} is not a live allocation (unknown or already freed)", base);
        return Error::InvalidPointer;
    }
    out = it->second.block;
    map_.erase(it);
    return Error::Success;
}

Error AllocationRegistry::checkRange(const void* ptr, size_t bytes, std::string& detail) const {
    if (bytes == 0) return Error::Success;
    if (!ptr) { detail = "null device pointer"; return Error::InvalidPointer; }
    std::shared_lock lk(mutex_);
    const auto key = reinterpret_cast<uintptr_t>(ptr);
    auto it = map_.upper_bound(key);
    if (it == map_.begin()) {
        detail = std::format("pointer {} is not inside any allocation", ptr);
        return Error::InvalidPointer;
    }
    --it;
    const size_t size = it->second.block.user_bytes;
    const uintptr_t offset = key - it->first;
    if (offset >= size) {
        detail = std::format("pointer {} is not inside any allocation", ptr);
        return Error::InvalidPointer;
    }
    if (bytes > size - offset) {
        detail = std::format("range [{}, +{}) exceeds allocation {} of {} bytes at offset {}", ptr, bytes,
                             static_cast<const void*>(it->second.block.user), size, offset);
        return Error::InvalidPointer;
    }
    return Error::Success;
}

size_t AllocationRegistry::count() const {
    std::shared_lock lk(mutex_);
    return map_.size();
}

std::vector<cpu::HostBlock> AllocationRegistry::drain() {
    std::unique_lock lk(mutex_);
    std::vector<cpu::HostBlock> v;
    v.reserve(map_.size());
    for (auto& [k, e] : map_) v.push_back(e.block);
    map_.clear();
    return v;
}

} // namespace kc
