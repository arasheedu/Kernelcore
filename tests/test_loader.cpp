// ELF validator and image loader: real Booth objects, targeted corruptions, random mutations.

#include "harness.hpp"
#include "test_kernels.hpp"

#include "cpu/booth_elf.hpp"
#include "cpu/booth_image.hpp"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using kc::Error;
using kc::cpu::BoothImage;
using kc::cpu::ElfImageInfo;
using kc::cpu::ElfLimits;
using kc::cpu::validateBoothElf;

namespace {

std::vector<unsigned char> bytesOf(const char* name) {
    for (const auto& r : testkernels::images())
        if (std::strcmp(r.name, name) == 0) return {r.data, r.data + r.size};
    return {};
}

ElfLimits limits() {
    ElfLimits l;
    l.allowed_externals = kc::cpu::defaultAllowedExternals();
    return l;
}

bool valid(const std::vector<unsigned char>& b, const char* kernel = "", std::string* why = nullptr) {
    ElfImageInfo info;
    std::string reason;
    const bool ok = validateBoothElf(b, kernel, limits(), info, reason);
    if (why) *why = reason;
    return ok;
}

// Replaces the first occurrence of `from` (same length as `to`) in the buffer.
bool patchString(std::vector<unsigned char>& b, const std::string& from, const std::string& to) {
    auto it = std::search(b.begin(), b.end(), from.begin(), from.end());
    if (it == b.end() || from.size() != to.size()) return false;
    std::copy(to.begin(), to.end(), it);
    return true;
}

template <class T>
T rd(const std::vector<unsigned char>& b, size_t off) {
    T v;
    std::memcpy(&v, b.data() + off, sizeof v);
    return v;
}
template <class T>
void wr(std::vector<unsigned char>& b, size_t off, T v) {
    std::memcpy(b.data() + off, &v, sizeof v);
}

// File offset of the section header whose type is `type`.
size_t shdrOfType(const std::vector<unsigned char>& b, uint32_t type) {
    const uint64_t shoff = rd<uint64_t>(b, 40);
    const uint16_t shnum = rd<uint16_t>(b, 60);
    for (uint16_t i = 0; i < shnum; ++i) {
        const size_t off = static_cast<size_t>(shoff) + i * 64u;
        if (rd<uint32_t>(b, off + 4) == type) return off;
    }
    return 0;
}

} // namespace

TEST_CASE(all_embedded_objects_validate) {
    for (const auto& r : testkernels::images()) {
        std::vector<unsigned char> b(r.data, r.data + r.size);
        ElfImageInfo info;
        std::string why;
        CHECK(validateBoothElf(b, r.name, limits(), info, why));
        if (!why.empty()) std::fprintf(stderr, "  %s: %s\n", r.name, why.c_str());
        CHECK(info.kernel_name == r.name);
        CHECK(kc::cpu::sha256Hex(b) == r.sha256);
    }
}

TEST_CASE(trig_mix_has_four_allowlisted_relocations) {
    const auto b = bytesOf("trig_mix");
    ElfImageInfo info;
    std::string why;
    REQUIRE(validateBoothElf(b, "trig_mix", limits(), info, why));
    CHECK(info.relocs.size() == 4);
    CHECK(info.externals.size() == 4);
}

TEST_CASE(deep_recursion_entry_is_not_at_offset_zero) {
    const auto b = bytesOf("deep_recursion");
    ElfImageInfo info;
    std::string why;
    REQUIRE(validateBoothElf(b, "deep_recursion", limits(), info, why));
    CHECK(info.entry_offset > 0);  // device helper comes first in .text
}

TEST_CASE(load_maps_relocates_and_unloads) {
    for (const auto& r : testkernels::images()) {
        std::unique_ptr<BoothImage> img;
        std::string why;
        CHECK_OK(BoothImage::load({r.data, r.size}, r.name, r.sha256, img, why));
        REQUIRE(img);
        CHECK(img->entry() >= img->base());
        MEMORY_BASIC_INFORMATION mbi{};
        REQUIRE(VirtualQuery(img->base(), &mbi, sizeof mbi) != 0);
        CHECK(mbi.Protect == PAGE_EXECUTE_READ);  // W^X: never writable once executable
    }
}

TEST_CASE(wrong_name_and_wrong_hash_are_rejected) {
    const auto b = bytesOf("vector_add");
    std::unique_ptr<BoothImage> img;
    std::string why;
    CHECK_ERR(BoothImage::load(b, "not_vector_add", "", img, why), Error::InvalidKernelImage);
    CHECK(why.find("expected 'not_vector_add'") != std::string::npos);
    CHECK_ERR(BoothImage::load(b, "vector_add", std::string(64, '0'), img, why), Error::InvalidKernelImage);
    CHECK(!img);
}

TEST_CASE(every_truncation_is_rejected) {
    for (const char* name : {"vector_add", "trig_mix"}) {
        const auto b = bytesOf(name);
        for (size_t n = 0; n < b.size(); ++n) {
            std::vector<unsigned char> t(b.begin(), b.begin() + static_cast<ptrdiff_t>(n));
            CHECK(!valid(t));
        }
    }
}

TEST_CASE(targeted_corruptions_are_rejected_with_reasons) {
    const auto base = bytesOf("trig_mix");
    REQUIRE(valid(base, "trig_mix"));
    struct Case {
        const char* what;
        void (*mutate)(std::vector<unsigned char>&);
        const char* reason_contains;
    };
    const Case cases[] = {
        {"magic", [](auto& b) { b[1] = 'X'; }, "magic"},
        {"class32", [](auto& b) { b[4] = 1; }, "ELFCLASS64"},
        {"big endian", [](auto& b) { b[5] = 2; }, "little-endian"},
        {"exec type", [](auto& b) { wr<uint16_t>(b, 16, 2); }, "ET_REL"},
        {"machine", [](auto& b) { wr<uint16_t>(b, 18, 3); }, "x86-64"},
        {"shstrndx", [](auto& b) { wr<uint16_t>(b, 62, 99); }, "shstrndx"},
        {"shoff huge", [](auto& b) { wr<uint64_t>(b, 40, ~0ull - 8); }, "outside file"},
        {"new section", [](auto& b) { patchString(b, ".strtab", ".rodata"); }, "unexpected section"},
        {"external not allowlisted", [](auto& b) { patchString(b, "exp2f", "systm"); }, "allowlist"},
        {"relocation type", [](auto& b) {
             const size_t sh = shdrOfType(b, 4);
             const size_t rel = static_cast<size_t>(rd<uint64_t>(b, sh + 24));
             wr<uint32_t>(b, rel + 8, 2);  // R_X86_64_PC32
         }, "relocation type"},
        {"relocation offset", [](auto& b) {
             const size_t sh = shdrOfType(b, 4);
             const size_t rel = static_cast<size_t>(rd<uint64_t>(b, sh + 24));
             wr<uint64_t>(b, rel, 0xFFFFFFFFull);
         }, "relocation outside"},
        {"relocation to kernel symbol", [](auto& b) {
             const size_t sh = shdrOfType(b, 4);
             const size_t rel = static_cast<size_t>(rd<uint64_t>(b, sh + 24));
             wr<uint64_t>(b, rel + 8, (1ull << 32) | 4u);
         }, "allowed external"},
        {"symbol name offset", [](auto& b) {
             const size_t sh = shdrOfType(b, 2);
             const size_t sym = static_cast<size_t>(rd<uint64_t>(b, sh + 24)) + 24;
             wr<uint32_t>(b, sym, 0x7FFFFFFF);
         }, "string offset"},
        {"local binding", [](auto& b) {
             const size_t sh = shdrOfType(b, 2);
             const size_t sym = static_cast<size_t>(rd<uint64_t>(b, sh + 24)) + 24;
             b[sym + 4] = 0x02;  // STB_LOCAL, STT_FUNC
         }, "non-global"},
        {"function past text", [](auto& b) {
             const size_t sh = shdrOfType(b, 2);
             const size_t sym = static_cast<size_t>(rd<uint64_t>(b, sh + 24)) + 24;
             wr<uint64_t>(b, sym + 16, 0x100000);  // st_size
         }, "outside .text"},
        {"writable text", [](auto& b) {
             const size_t sh = shdrOfType(b, 1);
             wr<uint64_t>(b, sh + 8, 0x7);
         }, "writable"},
        {"strtab unterminated", [](auto& b) {
             const size_t sh = shdrOfType(b, 2);
             const uint32_t strndx = rd<uint32_t>(b, sh + 40);
             const size_t ssh = static_cast<size_t>(rd<uint64_t>(b, 40)) + strndx * 64u;
             const size_t off = static_cast<size_t>(rd<uint64_t>(b, ssh + 24));
             const size_t sz = static_cast<size_t>(rd<uint64_t>(b, ssh + 32));
             b[off + sz - 1] = 'x';
         }, "NUL"},
    };
    for (const Case& c : cases) {
        auto b = base;
        c.mutate(b);
        std::string why;
        const bool ok = valid(b, "trig_mix", &why);
        if (ok || why.find(c.reason_contains) == std::string::npos)
            th::fail(__FILE__, __LINE__,
                     std::string("corruption '") + c.what + "': valid=" + (ok ? "true" : "false") + " reason='" +
                         why + "', expected reason containing '" + c.reason_contains + "'");
    }
}

TEST_CASE(random_mutations_never_crash_the_loader) {
    std::mt19937 rng(12345);
    int accepted = 0, rejected = 0;
    for (const char* name : {"vector_add", "trig_mix", "deep_recursion"}) {
        const auto base = bytesOf(name);
        REQUIRE(!base.empty());
        for (int iter = 0; iter < 20000; ++iter) {
            auto b = base;
            const int flips = 1 + static_cast<int>(rng() % 4);
            for (int f = 0; f < flips; ++f) b[rng() % b.size()] = static_cast<unsigned char>(rng());
            if (rng() % 16 == 0) b.resize(rng() % b.size());
            std::unique_ptr<BoothImage> img;
            std::string why;
            // No hash check, so mutations reach the parser and mapper. Never executed.
            const Error e = BoothImage::load(b, "", "", img, why);
            if (e == Error::Success) ++accepted;
            else if (e == Error::InvalidKernelImage) ++rejected;
            else th::fail(__FILE__, __LINE__, "unexpected load result " + th::describe(e) + ": " + why);
        }
    }
    std::printf("  mutations: %d accepted (code-byte changes), %d rejected\n", accepted, rejected);
    CHECK(rejected > 0);
}
