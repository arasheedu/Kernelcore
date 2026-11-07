# Booth kernel build integration.
#
#   booth_add_kernel(TARGET <tgt> NAME <symbol> SOURCE <file.cu>)
#
# compiles the kernel with kath --cpu (or takes a verified prebuilt object), validates
# it with pcuda_objcheck, and embeds it into <tgt> as pcuda_kimg_<symbol>.
#
# PCUDA_KATH_MODE selects how objects are produced:
#   auto      exe -> wsl -> prebuilt, first that works (default)
#   exe       pinned kath.exe (KATH_EXECUTABLE), SHA-256 checked
#   wsl       kath inside WSL (scripts/install_kath_wsl.sh)
#   prebuilt  kernels/prebuilt/<name>.o, bound to the source hash

include("${CMAKE_CURRENT_LIST_DIR}/BoothPin.cmake")
set(PCUDA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(PCUDA_PREBUILT_DIR "${PROJECT_SOURCE_DIR}/kernels/prebuilt")

set(PCUDA_KATH_MODE "auto" CACHE STRING "How Booth kernel objects are produced: auto|exe|wsl|prebuilt")
set_property(CACHE PCUDA_KATH_MODE PROPERTY STRINGS auto exe wsl prebuilt)
find_program(KATH_EXECUTABLE NAMES kath
  HINTS "${PROJECT_SOURCE_DIR}/third_party/booth-${BOOTH_VERSION}/dist/booth-${BOOTH_VERSION}-windows-x86_64"
  DOC "Booth kath.exe (pinned ${BOOTH_VERSION})")

function(_pcuda_probe_exe ok_var why_var)
  set(${ok_var} FALSE PARENT_SCOPE)
  if(NOT KATH_EXECUTABLE)
    set(${why_var} "exe: kath.exe not found (set KATH_EXECUTABLE or run scripts/bootstrap.ps1 -FetchKath)" PARENT_SCOPE)
    return()
  endif()
  file(SHA256 "${KATH_EXECUTABLE}" h)
  if(NOT h STREQUAL BOOTH_WIN_KATH_SHA256)
    set(${why_var} "exe: ${KATH_EXECUTABLE} SHA-256 ${h} does not match pinned ${BOOTH_WIN_KATH_SHA256}" PARENT_SCOPE)
    return()
  endif()
  execute_process(COMMAND "${KATH_EXECUTABLE}" --version
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 60)
  if(NOT rc EQUAL 0 OR NOT out MATCHES "Booth ${BOOTH_VERSION}")
    set(${why_var} "exe: '${KATH_EXECUTABLE} --version' failed or is not Booth ${BOOTH_VERSION}" PARENT_SCOPE)
    return()
  endif()
  set(${ok_var} TRUE PARENT_SCOPE)
endfunction()

function(_pcuda_probe_wsl ok_var why_var)
  set(${ok_var} FALSE PARENT_SCOPE)
  find_program(PCUDA_WSL_EXE wsl)
  if(NOT PCUDA_WSL_EXE)
    set(${why_var} "wsl: wsl.exe not found" PARENT_SCOPE)
    return()
  endif()
  execute_process(COMMAND "${PCUDA_WSL_EXE}" -e sh -c "${BOOTH_WSL_KATH_PATH} --version"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 180)
  if(NOT rc EQUAL 0 OR NOT out MATCHES "Booth ${BOOTH_VERSION}")
    set(${why_var} "wsl: ${BOOTH_WSL_KATH_PATH} not usable (run scripts/install_kath_wsl.sh inside WSL)" PARENT_SCOPE)
    return()
  endif()
  set(${ok_var} TRUE PARENT_SCOPE)
endfunction()

function(_pcuda_probe_prebuilt ok_var why_var)
  file(GLOB objs "${PCUDA_PREBUILT_DIR}/*.o")
  if(objs)
    set(${ok_var} TRUE PARENT_SCOPE)
  else()
    set(${ok_var} FALSE PARENT_SCOPE)
    set(${why_var} "prebuilt: no objects in ${PCUDA_PREBUILT_DIR}" PARENT_SCOPE)
  endif()
endfunction()

set(_mode "${PCUDA_KATH_MODE}")
set(_reasons "")
if(_mode STREQUAL "auto")
  set(PCUDA_KATH_RESOLVED "")
  foreach(m exe wsl prebuilt)
    cmake_language(CALL _pcuda_probe_${m} _ok _why)
    if(_ok)
      set(PCUDA_KATH_RESOLVED "${m}")
      break()
    endif()
    list(APPEND _reasons "${_why}")
  endforeach()
  if(NOT PCUDA_KATH_RESOLVED)
    list(JOIN _reasons "\n  " _r)
    message(FATAL_ERROR "pcuda: no way to produce Booth kernel objects:\n  ${_r}")
  endif()
  foreach(_r IN LISTS _reasons)
    message(STATUS "pcuda: skipped ${_r}")
  endforeach()
elseif(_mode MATCHES "^(exe|wsl|prebuilt)$")
  cmake_language(CALL _pcuda_probe_${_mode} _ok _why)
  if(NOT _ok)
    message(FATAL_ERROR "pcuda: PCUDA_KATH_MODE=${_mode} is not usable: ${_why}")
  endif()
  set(PCUDA_KATH_RESOLVED "${_mode}")
else()
  message(FATAL_ERROR "pcuda: PCUDA_KATH_MODE must be auto, exe, wsl or prebuilt (got '${_mode}')")
endif()
message(STATUS "pcuda: Booth ${BOOTH_VERSION} kernel objects via '${PCUDA_KATH_RESOLVED}'")

function(booth_add_kernel)
  cmake_parse_arguments(BK "" "TARGET;NAME;SOURCE" "" ${ARGN})
  if(NOT BK_TARGET OR NOT BK_NAME OR NOT BK_SOURCE)
    message(FATAL_ERROR "booth_add_kernel: TARGET, NAME and SOURCE are required")
  endif()
  get_filename_component(src "${BK_SOURCE}" ABSOLUTE)
  set(work "${PROJECT_BINARY_DIR}/kernels")
  set(obj "${work}/${BK_NAME}.o")
  set(cpp "${work}/${BK_NAME}_image.cpp")
  set(deps "${src}" "${PCUDA_CMAKE_DIR}/KathCompile.cmake" "${PCUDA_CMAKE_DIR}/EmbedBinary.cmake")
  if(PCUDA_KATH_RESOLVED STREQUAL "prebuilt")
    list(APPEND deps "${PCUDA_PREBUILT_DIR}/${BK_NAME}.o" "${PCUDA_PREBUILT_DIR}/${BK_NAME}.o.sha256"
                     "${PCUDA_PREBUILT_DIR}/${BK_NAME}.cu.sha256")
  elseif(PCUDA_KATH_RESOLVED STREQUAL "exe")
    list(APPEND deps "${KATH_EXECUTABLE}")
  endif()
  add_custom_command(
    OUTPUT "${obj}" "${cpp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DMODE=${PCUDA_KATH_RESOLVED}" "-DKATH=${KATH_EXECUTABLE}" "-DWSL_KATH=${BOOTH_WSL_KATH_PATH}"
            "-DSOURCE=${src}" "-DOUTPUT=${obj}" "-DNAME=${BK_NAME}" "-DWORK_DIR=${work}"
            "-DPREBUILT_DIR=${PCUDA_PREBUILT_DIR}" "-DBOOTH_VERSION=${BOOTH_VERSION}"
            -P "${PCUDA_CMAKE_DIR}/KathCompile.cmake"
    COMMAND $<TARGET_FILE:pcuda_objcheck> "${obj}" --kernel "${BK_NAME}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${obj}" "-DOUTPUT_CPP=${cpp}" "-DNAME=${BK_NAME}"
            "-DPROVENANCE_FILE=${obj}.provenance" -P "${PCUDA_CMAKE_DIR}/EmbedBinary.cmake"
    DEPENDS ${deps} pcuda_objcheck
    COMMENT "Booth kernel ${BK_NAME} (${PCUDA_KATH_RESOLVED}): compile, validate, embed"
    VERBATIM)
  target_sources(${BK_TARGET} PRIVATE "${cpp}")
  set_property(GLOBAL APPEND PROPERTY PCUDA_KERNELS "${BK_NAME}")
  set_property(GLOBAL PROPERTY PCUDA_KERNEL_OBJ_${BK_NAME} "${obj}")

  # Native code paths: Booth's --metal output (+ --ir for exact semantics) -> pcuda_mslgen ->
  # C++ compiled by MSVC. <name>_native.cpp runs one thread at a time; <name>_simd.cpp runs
  # 8 threads per AVX2 register (or is a stub when the kernel cannot be translated).
  set(msl "${work}/${BK_NAME}.metal")
  set(ir "${work}/${BK_NAME}.ir")
  set(ncpp "${work}/${BK_NAME}_native.cpp")
  set(scpp "${work}/${BK_NAME}_simd.cpp")
  set(mdeps "${src}" "${PCUDA_CMAKE_DIR}/KathCompile.cmake")
  if(PCUDA_KATH_RESOLVED STREQUAL "prebuilt")
    foreach(a metal ir)
      list(APPEND mdeps "${PCUDA_PREBUILT_DIR}/${BK_NAME}.${a}" "${PCUDA_PREBUILT_DIR}/${BK_NAME}.${a}.sha256")
    endforeach()
    list(APPEND mdeps "${PCUDA_PREBUILT_DIR}/${BK_NAME}.cu.sha256")
  elseif(PCUDA_KATH_RESOLVED STREQUAL "exe")
    list(APPEND mdeps "${KATH_EXECUTABLE}")
  endif()
  add_custom_command(
    OUTPUT "${msl}" "${ir}" "${ncpp}" "${scpp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DMODE=${PCUDA_KATH_RESOLVED}" "-DKATH=${KATH_EXECUTABLE}" "-DWSL_KATH=${BOOTH_WSL_KATH_PATH}"
            "-DSOURCE=${src}" "-DOUTPUT=${msl}" "-DOUTPUT_IR=${ir}" "-DEMIT=metal" "-DNAME=${BK_NAME}"
            "-DWORK_DIR=${work}" "-DPREBUILT_DIR=${PCUDA_PREBUILT_DIR}" "-DBOOTH_VERSION=${BOOTH_VERSION}"
            -P "${PCUDA_CMAKE_DIR}/KathCompile.cmake"
    COMMAND $<TARGET_FILE:pcuda_mslgen> --metal "${msl}" --ir "${ir}" --kernel "${BK_NAME}"
            --native "${ncpp}" --simd "${scpp}"
    DEPENDS ${mdeps} pcuda_mslgen
    COMMENT "Booth kernel ${BK_NAME} (${PCUDA_KATH_RESOLVED}): --metal/--ir -> native and simd C++"
    VERBATIM)
  target_sources(${BK_TARGET} PRIVATE "${ncpp}" "${scpp}")
  set_source_files_properties("${ncpp}" "${scpp}" TARGET_DIRECTORY ${BK_TARGET} PROPERTIES
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src/cpu/msl;${PROJECT_SOURCE_DIR}/include")
  set_source_files_properties("${ncpp}" TARGET_DIRECTORY ${BK_TARGET} PROPERTIES COMPILE_OPTIONS "/wd5030;/wd5222")
  set_source_files_properties("${scpp}" TARGET_DIRECTORY ${BK_TARGET} PROPERTIES COMPILE_OPTIONS "/wd5030;/wd5222;/arch:AVX2")
  set_property(GLOBAL PROPERTY PCUDA_KERNEL_MSL_${BK_NAME} "${msl}")
  set_property(GLOBAL PROPERTY PCUDA_KERNEL_IR_${BK_NAME} "${ir}")
endfunction()

# Target refresh_prebuilt: copy freshly compiled kath output (exe/wsl mode) into kernels/prebuilt.
function(pcuda_add_refresh_prebuilt_target)
  get_property(names GLOBAL PROPERTY PCUDA_KERNELS)
  set(cmds "")
  set(objs "")
  foreach(n IN LISTS names)
    get_property(o GLOBAL PROPERTY PCUDA_KERNEL_OBJ_${n})
    get_property(msl GLOBAL PROPERTY PCUDA_KERNEL_MSL_${n})
    get_property(ir GLOBAL PROPERTY PCUDA_KERNEL_IR_${n})
    list(APPEND objs "${o}" "${msl}" "${ir}")
    list(APPEND cmds COMMAND "${CMAKE_COMMAND}" "-DOBJ=${o}" "-DMSL=${msl}" "-DIR=${ir}" "-DNAME=${n}"
                             "-DDEST=${PCUDA_PREBUILT_DIR}" -P "${PCUDA_CMAKE_DIR}/RefreshPrebuilt.cmake")
  endforeach()
  add_custom_target(refresh_prebuilt ${cmds} DEPENDS ${objs}
    COMMENT "Refreshing kernels/prebuilt from kath output" VERBATIM)
endfunction()
