#pragma once

#include <kernelcore/kernel_registry.hpp>

#include <vector>

namespace kc {

// Defined in adapters/builtin_kernels.cpp: kernels embedded into the runtime library.
void registerBuiltinKernels(std::vector<const KernelEntry*>& out);

} // namespace kc
