#pragma once

#include <pcuda/kernel_registry.hpp>

#include <vector>

namespace pcuda {

// Defined in adapters/builtin_kernels.cpp: kernels embedded into the runtime library.
void registerBuiltinKernels(std::vector<const KernelEntry*>& out);

} // namespace pcuda
