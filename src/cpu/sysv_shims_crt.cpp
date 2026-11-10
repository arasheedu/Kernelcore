// Plain Win64 functions called by the System V shims in sysv_shims.asm.
// Wrapping the CRT here guarantees real out-of-line symbols regardless of how
// the CRT headers declare sinf & co.

#include <cmath>

extern "C" {
float kc_crt_sinf(float x) noexcept { return std::sin(x); }
float kc_crt_cosf(float x) noexcept { return std::cos(x); }
float kc_crt_exp2f(float x) noexcept { return std::exp2(x); }
float kc_crt_log2f(float x) noexcept { return std::log2(x); }
}
