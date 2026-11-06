# Test kernels (Booth-compiled, test-only) and their adapters. Also used by pcuda_vs_cuda.
add_library(pcuda_test_kernels STATIC test_kernels.cpp)
target_link_libraries(pcuda_test_kernels PUBLIC pcuda)
target_include_directories(pcuda_test_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
foreach(k oob_write int_div deep_recursion scale_add trig_mix)
  booth_add_kernel(TARGET pcuda_test_kernels NAME ${k} SOURCE "${PROJECT_SOURCE_DIR}/kernels/test/${k}.cu")
endforeach()

add_library(pcuda_test_support STATIC harness_main.cpp sysv_test_funcs.asm)
target_link_libraries(pcuda_test_support PUBLIC pcuda_test_kernels)
target_include_directories(pcuda_test_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src")

# ASan runtime DLL lives next to cl.exe.
get_filename_component(_msvc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)

set(PCUDA_TESTS test_api test_memory test_loader test_thunk_abi test_vector_add test_fault test_stream
                test_chunked_vector_add test_llm_kernels test_codegen)
foreach(t IN LISTS PCUDA_TESTS)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE pcuda_test_support pcuda_llm)
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES TIMEOUT 600
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_msvc_bin}")
endforeach()

# Suites that run kernels also run on each code generation path (PCUDA_CODEGEN). The plain
# test names above use the default (auto = simd where available, else native).
foreach(t test_vector_add test_fault test_stream test_chunked_vector_add test_llm_kernels)
  foreach(cg booth native simd)
    add_test(NAME ${t}_${cg} COMMAND ${t})
    set_tests_properties(${t}_${cg} PROPERTIES TIMEOUT 600
      ENVIRONMENT "PCUDA_CODEGEN=${cg}"
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_msvc_bin}")
  endforeach()
endforeach()
