#include "booth_elf.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace kc::cpu {
namespace {

constexpr uint16_t kEtRel = 1;
constexpr uint16_t kEmX86_64 = 62;
constexpr uint32_t kShtNull = 0, kShtProgbits = 1, kShtSymtab = 2, kShtStrtab = 3, kShtRela = 4;
constexpr uint64_t kShfWrite = 0x1, kShfAlloc = 0x2, kShfExec = 0x4;
constexpr uint8_t kStbGlobal = 1;
constexpr uint8_t kSttNotype = 0, kSttFunc = 2;
constexpr uint32_t kRX86_64_PLT32 = 4;
constexpr uint16_t kShnUndef = 0;
constexpr uint64_t kEhdrSize = 64, kShdrSize = 64, kSymSize = 24, kRelaSize = 24;
constexpr uint16_t kMaxSections = 64;

struct Shdr {
    uint32_t name, type;
    uint64_t flags, addr, offset, size;
    uint32_t link, info;
    uint64_t addralign, entsize;
};

template <class T>
T rd(const unsigned char* p) noexcept {
    T v;
    std::memcpy(&v, p, sizeof(T));  // x64 is little-endian, matching ELFDATA2LSB
    return v;
}

bool addOk(uint64_t a, uint64_t b, uint64_t& r) noexcept {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    r = a + b;
    return true;
}

bool inFile(uint64_t off, uint64_t size, uint64_t file) noexcept {
    uint64_t end;
    return addOk(off, size, end) && end <= file;
}

class Parser {
public:
    Parser(std::span<const unsigned char> b, std::string& reason) : b_(b), reason_(reason) {}

    bool fail(std::string msg) {
        reason_ = std::move(msg);
        return false;
    }

    // NUL-terminated string at `off` inside a string-table section.
    bool str(const Shdr& tab, uint64_t off, std::string_view& out) {
        if (off >= tab.size) return fail("string offset out of range");
        const auto* base = reinterpret_cast<const char*>(b_.data() + tab.offset);
        const void* nul = std::memchr(base + off, 0, static_cast<size_t>(tab.size - off));
        if (!nul) return fail("unterminated string in string table");
        out = std::string_view(base + off, static_cast<size_t>(static_cast<const char*>(nul) - (base + off)));
        return true;
    }

    std::span<const unsigned char> b_;
    std::string& reason_;
};

} // namespace

std::span<const std::string_view> defaultAllowedExternals() noexcept {
    static constexpr std::array<std::string_view, 4> k{"sinf", "cosf", "exp2f", "log2f"};
    return k;
}

bool validateBoothElf(std::span<const unsigned char> bytes, std::string_view expected_kernel,
                      const ElfLimits& limits, ElfImageInfo& out, std::string& reason) {
    out = ElfImageInfo{};
    Parser P(bytes, reason);
    const uint64_t fsize = bytes.size();
    const unsigned char* d = bytes.data();

    // ---- ELF header ----
    if (fsize < kEhdrSize) return P.fail("file smaller than ELF header");
    if (!(d[0] == 0x7f && d[1] == 'E' && d[2] == 'L' && d[3] == 'F')) return P.fail("bad ELF magic");
    if (d[4] != 2) return P.fail("not ELFCLASS64");
    if (d[5] != 1) return P.fail("not little-endian");
    if (d[6] != 1) return P.fail("bad EI_VERSION");
    if (rd<uint16_t>(d + 16) != kEtRel) return P.fail("not a relocatable object (ET_REL)");
    if (rd<uint16_t>(d + 18) != kEmX86_64) return P.fail("machine is not x86-64");
    if (rd<uint32_t>(d + 20) != 1) return P.fail("bad e_version");
    const uint64_t shoff = rd<uint64_t>(d + 40);
    if (rd<uint16_t>(d + 52) != kEhdrSize) return P.fail("bad e_ehsize");
    if (rd<uint16_t>(d + 56) != 0) return P.fail("relocatable object has program headers");
    if (rd<uint16_t>(d + 58) != kShdrSize) return P.fail("bad e_shentsize");
    const uint16_t shnum = rd<uint16_t>(d + 60);
    const uint16_t shstrndx = rd<uint16_t>(d + 62);
    if (shnum < 2 || shnum > kMaxSections) return P.fail("unexpected section count");
    if (shstrndx == 0 || shstrndx >= shnum) return P.fail("bad e_shstrndx");
    if (!inFile(shoff, uint64_t{shnum} * kShdrSize, fsize)) return P.fail("section headers outside file");

    // ---- section headers ----
    std::vector<Shdr> sh(shnum);
    for (uint16_t i = 0; i < shnum; ++i) {
        const unsigned char* p = d + shoff + uint64_t{i} * kShdrSize;
        Shdr& s = sh[i];
        s.name = rd<uint32_t>(p + 0);
        s.type = rd<uint32_t>(p + 4);
        s.flags = rd<uint64_t>(p + 8);
        s.addr = rd<uint64_t>(p + 16);
        s.offset = rd<uint64_t>(p + 24);
        s.size = rd<uint64_t>(p + 32);
        s.link = rd<uint32_t>(p + 40);
        s.info = rd<uint32_t>(p + 44);
        s.addralign = rd<uint64_t>(p + 48);
        s.entsize = rd<uint64_t>(p + 56);
        if (i != 0 && !inFile(s.offset, s.size, fsize)) return P.fail("section data outside file");
    }
    if (sh[0].type != kShtNull) return P.fail("section 0 is not SHT_NULL");
    const Shdr& shstr = sh[shstrndx];
    if (shstr.type != kShtStrtab || shstr.size == 0) return P.fail("bad section name table");

    int itext = -1, irela = -1, isym = -1, istr = -1;
    for (uint16_t i = 1; i < shnum; ++i) {
        std::string_view nm;
        if (!P.str(shstr, sh[i].name, nm)) return false;
        int* slot = nullptr;
        uint32_t want = 0;
        if (nm == ".text") { slot = &itext; want = kShtProgbits; }
        else if (nm == ".rela.text") { slot = &irela; want = kShtRela; }
        else if (nm == ".symtab") { slot = &isym; want = kShtSymtab; }
        else if (nm == ".strtab") { slot = &istr; want = kShtStrtab; }
        else if (nm == ".shstrtab") { if (i != shstrndx) return P.fail("stray .shstrtab"); continue; }
        else return P.fail("unexpected section '" + std::string(nm) + "' (Booth output format changed?)");
        if (*slot != -1) return P.fail("duplicate section '" + std::string(nm) + "'");
        if (sh[i].type != want) return P.fail("section '" + std::string(nm) + "' has wrong type");
        *slot = i;
    }
    if (itext < 0 || irela < 0 || isym < 0 || istr < 0) return P.fail("missing required section");

    const Shdr& text = sh[itext];
    const Shdr& rela = sh[irela];
    const Shdr& symtab = sh[isym];
    const Shdr& strtab = sh[istr];

    if ((text.flags & (kShfAlloc | kShfExec)) != (kShfAlloc | kShfExec)) return P.fail(".text is not alloc+exec");
    if (text.flags & kShfWrite) return P.fail(".text is writable");
    if (text.size == 0) return P.fail(".text is empty");
    if (text.size > limits.max_text_bytes) return P.fail(".text exceeds size limit");
    if (symtab.entsize != kSymSize || symtab.size % kSymSize != 0) return P.fail("bad .symtab entry size");
    if (symtab.link != static_cast<uint32_t>(istr)) return P.fail(".symtab not linked to .strtab");
    if (rela.entsize != kRelaSize || rela.size % kRelaSize != 0) return P.fail("bad .rela.text entry size");
    if (rela.link != static_cast<uint32_t>(isym)) return P.fail(".rela.text not linked to .symtab");
    if (rela.info != static_cast<uint32_t>(itext)) return P.fail(".rela.text does not apply to .text");
    if (strtab.size == 0 || d[strtab.offset + strtab.size - 1] != 0) return P.fail(".strtab not NUL-terminated");

    out.text_offset = text.offset;
    out.text_size = text.size;

    // ---- symbols ----
    const uint64_t nsyms = symtab.size / kSymSize;
    if (nsyms < 2) return P.fail("no symbols");
    if (nsyms > limits.max_symbols) return P.fail("too many symbols");
    std::vector<int64_t> sym_to_ext(static_cast<size_t>(nsyms), -1);
    int defined = 0;
    for (uint64_t i = 1; i < nsyms; ++i) {
        const unsigned char* p = d + symtab.offset + i * kSymSize;
        const uint32_t st_name = rd<uint32_t>(p + 0);
        const uint8_t st_info = p[4];
        const uint16_t st_shndx = rd<uint16_t>(p + 6);
        const uint64_t st_value = rd<uint64_t>(p + 8);
        const uint64_t st_size = rd<uint64_t>(p + 16);
        const uint8_t bind = st_info >> 4, type = st_info & 0xf;
        std::string_view nm;
        if (!P.str(strtab, st_name, nm)) return false;
        if (nm.empty()) return P.fail("unnamed symbol");
        if (bind != kStbGlobal) return P.fail("unexpected non-global symbol '" + std::string(nm) + "'");

        if (st_shndx == static_cast<uint16_t>(itext) && type == kSttFunc) {
            uint64_t end;
            if (st_value >= text.size || !addOk(st_value, st_size, end) || end > text.size)
                return P.fail("function symbol outside .text");
            ++defined;
            out.kernel_name = std::string(nm);
            out.entry_offset = st_value;
            out.entry_size = st_size;
        } else if (st_shndx == kShnUndef && (type == kSttNotype || type == kSttFunc)) {
            const auto allowed = limits.allowed_externals;
            if (std::find(allowed.begin(), allowed.end(), nm) == allowed.end())
                return P.fail("external symbol '" + std::string(nm) + "' is not on the allowlist");
            sym_to_ext[static_cast<size_t>(i)] = static_cast<int64_t>(out.externals.size());
            out.externals.emplace_back(nm);
        } else {
            return P.fail("unsupported symbol kind for '" + std::string(nm) + "'");
        }
    }
    if (defined != 1) return P.fail("expected exactly one defined kernel function, found " + std::to_string(defined));
    if (!expected_kernel.empty() && out.kernel_name != expected_kernel)
        return P.fail("kernel symbol is '" + out.kernel_name + "', expected '" + std::string(expected_kernel) + "'");

    // ---- relocations ----
    const uint64_t nrel = rela.size / kRelaSize;
    if (nrel > limits.max_relocs) return P.fail("too many relocations");
    out.relocs.reserve(static_cast<size_t>(nrel));
    for (uint64_t i = 0; i < nrel; ++i) {
        const unsigned char* p = d + rela.offset + i * kRelaSize;
        const uint64_t r_offset = rd<uint64_t>(p + 0);
        const uint64_t r_info = rd<uint64_t>(p + 8);
        const int64_t r_addend = rd<int64_t>(p + 16);
        const uint32_t type = static_cast<uint32_t>(r_info & 0xffffffffu);
        const uint64_t sym = r_info >> 32;
        if (type != kRX86_64_PLT32) return P.fail("unsupported relocation type " + std::to_string(type));
        if (sym == 0 || sym >= nsyms || sym_to_ext[static_cast<size_t>(sym)] < 0)
            return P.fail("relocation does not target an allowed external");
        if (!inFile(r_offset, 4, text.size)) return P.fail("relocation outside .text");
        if (r_addend < std::numeric_limits<int32_t>::min() || r_addend > std::numeric_limits<int32_t>::max())
            return P.fail("relocation addend out of range");
        out.relocs.push_back({r_offset, r_addend, static_cast<uint32_t>(sym_to_ext[static_cast<size_t>(sym)])});
    }
    std::vector<uint64_t> offs;
    offs.reserve(out.relocs.size());
    for (const auto& r : out.relocs) offs.push_back(r.offset);
    std::sort(offs.begin(), offs.end());
    for (size_t i = 1; i < offs.size(); ++i)
        if (offs[i] - offs[i - 1] < 4) return P.fail("overlapping relocations");

    reason.clear();
    return true;
}

} // namespace kc::cpu
