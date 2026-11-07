# pcuda backend: Q4_K / Q6_K matrix multiplications as CUDA C kernels on the CPU cores.
# Builds the pcuda runtime and its LLM kernels from a CudaOnCPUCores checkout (GGML_PCUDA_ROOT),
# using the kernel objects prebuilt in that tree (no Booth compiler needed).

if (NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "GGML_PCUDA: Windows x64 with MSVC only")
endif()
if (NOT GGML_PCUDA_ROOT OR NOT EXISTS "${GGML_PCUDA_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "GGML_PCUDA: set GGML_PCUDA_ROOT to a CudaOnCPUCores checkout")
endif()

set(PCUDA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCUDA_KATH_MODE prebuilt CACHE STRING "" FORCE)
add_subdirectory(${GGML_PCUDA_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/pcuda EXCLUDE_FROM_ALL)

ggml_add_backend_library(ggml-pcuda ggml-pcuda.cpp)
set_target_properties(ggml-pcuda PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_link_libraries(ggml-pcuda PRIVATE pcuda pcuda_llm)
target_include_directories(ggml-pcuda PRIVATE ${GGML_PCUDA_ROOT}/llm ${GGML_PCUDA_ROOT}/include)
