#pragma once

// Maps a validated Booth ELF object into executable memory (W^X) on Windows x64.

#include "booth_elf.hpp"

#include <pcuda/error.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace pcuda::cpu {

class BoothImage {
public:
    BoothImage(const BoothImage&) = delete;
    BoothImage& operator=(const BoothImage&) = delete;
    ~BoothImage();

    // Validates, checks SHA-256 (if expected_sha256 is non-empty), maps, relocates and
    // makes the code read+execute. On failure returns an error and sets `detail`.
    static Error load(std::span<const unsigned char> bytes, std::string_view kernel_name,
                      std::string_view expected_sha256, std::unique_ptr<BoothImage>& out,
                      std::string& detail);

    const void*        entry() const noexcept { return entry_; }
    const void*        base() const noexcept { return base_; }
    size_t             size() const noexcept { return size_; }
    const std::string& name() const noexcept { return info_.kernel_name; }
    const ElfImageInfo& info() const noexcept { return info_; }

private:
    BoothImage() = default;
    void*        base_ = nullptr;
    size_t       size_ = 0;
    const void*  entry_ = nullptr;
    ElfImageInfo info_;
};

// Lowercase hex SHA-256 via Windows CNG. Returns empty string on failure.
std::string sha256Hex(std::span<const unsigned char> bytes);

// Address of the System V-callable shim for an allowlisted external, or nullptr.
const void* sysvShimFor(std::string_view external) noexcept;

} // namespace pcuda::cpu
