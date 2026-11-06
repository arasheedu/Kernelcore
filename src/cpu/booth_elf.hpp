#pragma once

// Strict validator for the ELF64 subset that Booth `kath --cpu` emits (verified for 0.6.1):
//   sections: NULL, .text, .rela.text, .shstrtab, .symtab, .strtab
//   one defined global function (the kernel), undefined externals from an allowlist,
//   relocations of type R_X86_64_PLT32 only.
// Anything else is rejected. Treats the input as untrusted: every offset is bounds-checked.
// Portable C++ (no Windows headers) so the same code runs in the build-time checker.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pcuda::cpu {

struct ElfReloc {
    uint64_t offset = 0;     // into .text; 4 bytes patched
    int64_t  addend = 0;
    uint32_t external = 0;   // index into ElfImageInfo::externals
};

struct ElfImageInfo {
    uint64_t                 text_offset = 0;  // file offset of .text
    uint64_t                 text_size = 0;
    uint64_t                 entry_offset = 0; // kernel symbol value, offset into .text
    uint64_t                 entry_size = 0;
    std::string              kernel_name;
    std::vector<std::string> externals;
    std::vector<ElfReloc>    relocs;
};

struct ElfLimits {
    std::span<const std::string_view> allowed_externals;
    uint64_t max_text_bytes = 16u * 1024u * 1024u;
    uint64_t max_relocs = 65536;
    uint64_t max_symbols = 65536;
};

// The externals Booth 0.6.1 may reference (src/cpu/cpu_emit.c).
std::span<const std::string_view> defaultAllowedExternals() noexcept;

// expected_kernel empty: accept the single defined function whatever its name.
// Returns false and sets `reason` on any violation. Never throws on malformed input
// (may throw std::bad_alloc).
bool validateBoothElf(std::span<const unsigned char> bytes, std::string_view expected_kernel,
                      const ElfLimits& limits, ElfImageInfo& out, std::string& reason);

} // namespace pcuda::cpu
