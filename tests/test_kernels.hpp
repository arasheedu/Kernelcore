#pragma once

// Test-only kernels compiled by Booth (kernels/test/*.cu) and their typed adapters.

#include <pcuda/api.hpp>

#include <span>

namespace testkernels {

struct OobWriteArgs {      // out[i + count] = 1 : writes past the end on purpose
    float* out;
    int    count;
};
struct IntDivArgs {        // out[i] = a[i] / divisor
    const int* a;
    int*       out;
    int        divisor;
    int        count;
};
struct DeepRecursionArgs { // out[i] = deep(depth) (recursive device function)
    int* out;
    int  depth;
    int  count;
};
struct ScaleAddArgs {      // out[i] = x[i] * scale + bias (float scalars in xmm)
    const float* x;
    float*       out;
    float        scale;
    float        bias;
    int          count;
};
struct TrigMixArgs {       // out[i] = sinf(x) + cosf(x) + exp2f(x) + log2f(x + 2)
    const float* x;
    float*       out;
    int          count;
};

pcuda::Error registerAll();

// All embedded kernel entries (including the runtime's vector_add), for loader tests.
struct ImageRef {
    const char*          name;
    const unsigned char* data;
    size_t               size;
    const char*          sha256;
};
std::span<const ImageRef> images();

} // namespace testkernels
