#pragma once

#include <string>

namespace pcuda::detail {

// Thread-local diagnostic text for the last failing call on this thread.
void               setDetail(std::string text) noexcept;
void               clearDetail() noexcept;
const std::string& lastDetail() noexcept;

// Runtime-owned threads (stream and pool workers) must not re-enter the public API:
// shutdown would deadlock waiting for them to drain.
void markRuntimeThread() noexcept;
bool isRuntimeThread() noexcept;

} // namespace pcuda::detail
