#include "harness.hpp"
#include "test_kernels.hpp"

#include <cstring>

RuntimeFixture::RuntimeFixture() {
    pcuda::Error e = pcudaInit();
    if (e != pcuda::Error::Success) {
        th::fail(__FILE__, __LINE__, "pcudaInit -> " + th::describe(e));
        return;
    }
    e = testkernels::registerAll();
    if (e != pcuda::Error::Success) {
        th::fail(__FILE__, __LINE__, "register test kernels -> " + th::describe(e));
        return;
    }
    ok = true;
}

RuntimeFixture::~RuntimeFixture() {
    const pcuda::Error e = pcudaShutdown();
    if (e != pcuda::Error::Success) th::fail(__FILE__, __LINE__, "pcudaShutdown -> " + th::describe(e));
}

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran = 0;
    for (const auto& c : th::cases()) {
        if (filter && !std::strstr(c.name, filter)) continue;
        const int before = th::failures();
        std::printf("[ RUN  ] %s\n", c.name);
        std::fflush(stdout);
        c.fn();
        std::printf("[ %s ] %s\n", th::failures() == before ? " OK " : "FAIL", c.name);
        std::fflush(stdout);
        ++ran;
    }
    std::printf("%d test case(s), %d failure(s)\n", ran, th::failures());
    if (ran == 0) return 2;
    return th::failures() == 0 ? 0 : 1;
}
