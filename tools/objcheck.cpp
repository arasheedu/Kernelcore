// kc_objcheck: build-time validation of a Booth --cpu object with the same parser
// the runtime uses. A bad object fails the build instead of reaching runtime.
//
// usage: kc_objcheck <object.o> [--kernel <name>]
// exit:  0 valid, 1 invalid, 2 usage or I/O error

#include "../src/cpu/booth_elf.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: kc_objcheck <object.o> [--kernel <name>]\n");
        return 2;
    }
    std::string path = argv[1], kernel;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--kernel" && i + 1 < argc) kernel = argv[++i];
        else {
            std::fprintf(stderr, "kc_objcheck: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "kc_objcheck: cannot open '%s'\n", path.c_str());
        return 2;
    }
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    kc::cpu::ElfLimits limits;
    limits.allowed_externals = kc::cpu::defaultAllowedExternals();
    kc::cpu::ElfImageInfo info;
    std::string reason;
    try {
        if (!kc::cpu::validateBoothElf(bytes, kernel, limits, info, reason)) {
            std::fprintf(stderr, "kc_objcheck: %s: INVALID: %s\n", path.c_str(), reason.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kc_objcheck: %s: error: %s\n", path.c_str(), e.what());
        return 2;
    }
    std::printf("kc_objcheck: %s: OK kernel=%s text=%llu bytes relocs=%zu externals=[", path.c_str(),
                info.kernel_name.c_str(), static_cast<unsigned long long>(info.text_size), info.relocs.size());
    for (size_t i = 0; i < info.externals.size(); ++i)
        std::printf("%s%s", i ? "," : "", info.externals[i].c_str());
    std::printf("]\n");
    return 0;
}
