# Kernelcore backend: Q4_K / Q6_K matrix multiplications as CUDA C kernels on the CPU cores.
# Builds the Kernelcore runtime and its LLM kernels from a Kernelcore checkout (GGML_KERNELCORE_ROOT),
# using the kernel objects prebuilt in that tree (no Booth compiler needed).

if (NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "GGML_KERNELCORE: Windows x64 with MSVC only")
endif()
if (NOT GGML_KERNELCORE_ROOT OR NOT EXISTS "${GGML_KERNELCORE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "GGML_KERNELCORE: set GGML_KERNELCORE_ROOT to a Kernelcore checkout")
endif()

set(KC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KC_KATH_MODE prebuilt CACHE STRING "" FORCE)
add_subdirectory(${GGML_KERNELCORE_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/kernelcore EXCLUDE_FROM_ALL)

ggml_add_backend_library(ggml-kernelcore ggml-kernelcore.cpp)
set_target_properties(ggml-kernelcore PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(ggml-kernelcore PRIVATE kernelcore kc_llm)
target_include_directories(ggml-kernelcore PRIVATE ${GGML_KERNELCORE_ROOT}/llm ${GGML_KERNELCORE_ROOT}/include)
