#pragma once

// Minimal in-repo test harness (no network fetch, no third-party dependency).

#include <kernelcore/api.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace th {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> v;
    return v;
}
inline int& failures() {
    static int f = 0;
    return f;
}
struct Reg {
    Reg(const char* n, void (*f)()) { cases().push_back({n, f}); }
};
inline void fail(const char* file, int line, const std::string& msg) {
    ++failures();
    std::fprintf(stderr, "%s(%d): FAILED: %s\n", file, line, msg.c_str());
    std::fflush(stderr);
}
inline std::string describe(kc::Error e) {
    std::string s = kcGetErrorString(e);
    const char* d = kcGetLastErrorDetail();
    if (d && *d) s += std::string(" (") + d + ")";
    return s;
}

} // namespace th

#define TEST_CASE(name)                                  \
    static void name();                                  \
    static ::th::Reg name##_registration(#name, &name);  \
    static void name()

#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) ::th::fail(__FILE__, __LINE__, "CHECK(" #c ")"); \
    } while (0)

#define REQUIRE(c)                                                    \
    do {                                                              \
        if (!(c)) {                                                   \
            ::th::fail(__FILE__, __LINE__, "REQUIRE(" #c ")");        \
            return;                                                   \
        }                                                             \
    } while (0)

#define CHECK_ERR(expr, want)                                                                         \
    do {                                                                                              \
        const ::kc::Error th_e_ = (expr);                                                          \
        if (th_e_ != (want))                                                                          \
            ::th::fail(__FILE__, __LINE__,                                                            \
                       std::string(#expr " -> ") + ::th::describe(th_e_) + ", expected " +            \
                           kcGetErrorString(want));                                                \
    } while (0)

#define CHECK_OK(expr) CHECK_ERR(expr, ::kc::Error::Success)

#define REQUIRE_OK(expr)                                                                              \
    do {                                                                                              \
        const ::kc::Error th_e_ = (expr);                                                          \
        if (th_e_ != ::kc::Error::Success) {                                                       \
            ::th::fail(__FILE__, __LINE__, std::string(#expr " -> ") + ::th::describe(th_e_));        \
            return;                                                                                   \
        }                                                                                             \
    } while (0)

// RAII: kcInit + test kernel registration, kcShutdown at scope exit.
struct RuntimeFixture {
    RuntimeFixture();
    ~RuntimeFixture();
    bool ok = false;
};
