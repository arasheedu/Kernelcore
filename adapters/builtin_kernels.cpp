// Explicit list of kernels embedded in the runtime library. Explicit (not static
// self-registration) so the linker can't drop an adapter and init order is fixed.

#include "../src/builtin_kernels.hpp"

namespace kc::kernels {
const KernelEntry& vectorAddEntry() noexcept;
}

namespace kc {

void registerBuiltinKernels(std::vector<const KernelEntry*>& out) {
    out.push_back(&kernels::vectorAddEntry());
}

} // namespace kc
