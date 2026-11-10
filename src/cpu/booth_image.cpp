#include "booth_image.hpp"

#include <windows.h>
#include <bcrypt.h>

#include <cstring>
#include <limits>

extern "C" {
// System V -> Win64 shims (sysv_shims.asm)
void kc_sysv_sinf();
void kc_sysv_cosf();
void kc_sysv_exp2f();
void kc_sysv_log2f();
}

namespace kc::cpu {
namespace {

constexpr size_t kStubSize = 16;  // FF 25 00000000 <abs64> + 2 x int3

size_t roundUp(size_t v, size_t a) noexcept { return (v + a - 1) / a * a; }

size_t pageSize() noexcept {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

} // namespace

const void* sysvShimFor(std::string_view n) noexcept {
    if (n == "sinf") return reinterpret_cast<const void*>(&kc_sysv_sinf);
    if (n == "cosf") return reinterpret_cast<const void*>(&kc_sysv_cosf);
    if (n == "exp2f") return reinterpret_cast<const void*>(&kc_sysv_exp2f);
    if (n == "log2f") return reinterpret_cast<const void*>(&kc_sysv_log2f);
    return nullptr;
}

std::string sha256Hex(std::span<const unsigned char> bytes) {
    unsigned char digest[32];
    if (bytes.size() > std::numeric_limits<ULONG>::max()) return {};
    const NTSTATUS st = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                                   const_cast<PUCHAR>(bytes.data()), static_cast<ULONG>(bytes.size()),
                                   digest, sizeof digest);
    if (!BCRYPT_SUCCESS(st)) return {};
    static constexpr char hex[] = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[2 * i] = hex[digest[i] >> 4];
        s[2 * i + 1] = hex[digest[i] & 15];
    }
    return s;
}

BoothImage::~BoothImage() {
    if (base_) VirtualFree(base_, 0, MEM_RELEASE);
}

Error BoothImage::load(std::span<const unsigned char> bytes, std::string_view kernel_name,
                       std::string_view expected_sha256, std::unique_ptr<BoothImage>& out,
                       std::string& detail) {
    out.reset();
    if (!expected_sha256.empty()) {
        const std::string got = sha256Hex(bytes);
        if (got.empty()) { detail = "SHA-256 computation failed"; return Error::InternalError; }
        if (got != expected_sha256) {
            detail = "kernel image SHA-256 mismatch (embedded data corrupted)";
            return Error::InvalidKernelImage;
        }
    }

    ElfLimits limits;
    limits.allowed_externals = defaultAllowedExternals();
    std::unique_ptr<BoothImage> img(new BoothImage());
    std::string why;
    if (!validateBoothElf(bytes, kernel_name, limits, img->info_, why)) {
        detail = "invalid Booth object: " + why;
        return Error::InvalidKernelImage;
    }
    const ElfImageInfo& info = img->info_;

    // Layout: [.text][pad to 16][one 16-byte jump stub per external], rounded to a page.
    const size_t text = static_cast<size_t>(info.text_size);
    const size_t stubs_off = roundUp(text, 16);
    const size_t total = roundUp(stubs_off + info.externals.size() * kStubSize, pageSize());

    void* mem = VirtualAlloc(nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!mem) { detail = "VirtualAlloc failed for kernel image"; return Error::OutOfMemory; }
    img->base_ = mem;
    img->size_ = total;
    auto* base = static_cast<unsigned char*>(mem);

    std::memset(base, 0xCC, total);  // int3 everywhere not covered by code
    std::memcpy(base, bytes.data() + info.text_offset, text);

    for (size_t i = 0; i < info.externals.size(); ++i) {
        const void* target = sysvShimFor(info.externals[i]);
        if (!target) { detail = "no shim for external '" + info.externals[i] + "'"; return Error::InvalidKernelImage; }
        unsigned char* stub = base + stubs_off + i * kStubSize;
        stub[0] = 0xFF; stub[1] = 0x25;                  // jmp qword ptr [rip+0]
        std::memset(stub + 2, 0, 4);
        const uint64_t abs = reinterpret_cast<uint64_t>(target);
        std::memcpy(stub + 6, &abs, 8);
    }

    for (const ElfReloc& r : info.relocs) {
        // R_X86_64_PLT32: S + A - P, where S is the external's local jump stub.
        const int64_t S = reinterpret_cast<int64_t>(base + stubs_off + r.external * kStubSize);
        const int64_t P = reinterpret_cast<int64_t>(base + r.offset);
        const int64_t v = S + r.addend - P;
        if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
            detail = "relocation does not fit in rel32";
            return Error::InvalidKernelImage;
        }
        const int32_t v32 = static_cast<int32_t>(v);
        std::memcpy(base + r.offset, &v32, 4);
    }

    DWORD old = 0;
    if (!VirtualProtect(mem, total, PAGE_EXECUTE_READ, &old)) {
        const DWORD e = GetLastError();
        if (e == ERROR_DYNAMIC_CODE_BLOCKED) {
            detail = "process policy (Arbitrary Code Guard) blocks executable memory";
            return Error::DynamicCodeBlocked;
        }
        detail = "VirtualProtect(PAGE_EXECUTE_READ) failed, error " + std::to_string(e);
        return Error::InternalError;
    }
    FlushInstructionCache(GetCurrentProcess(), mem, total);

    img->entry_ = base + info.entry_offset;
    out = std::move(img);
    return Error::Success;
}

} // namespace kc::cpu
