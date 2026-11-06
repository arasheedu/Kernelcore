# Build-time script (cmake -P): produce one Booth --cpu object.
# Inputs: MODE (exe|wsl|prebuilt), SOURCE, OUTPUT, NAME, WORK_DIR, PREBUILT_DIR,
#         KATH (exe mode), WSL_KATH (wsl mode), BOOTH_VERSION.
# Writes OUTPUT, OUTPUT.provenance and OUTPUT.srcsha256. Fails loudly on any problem.
cmake_minimum_required(VERSION 3.26)

foreach(v MODE SOURCE OUTPUT NAME WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "KathCompile: ${v} not set")
  endif()
endforeach()

# Normalise line endings so CRLF checkouts produce identical input (and source hashes).
file(READ "${SOURCE}" src)
string(REPLACE "\r\n" "\n" src "${src}")
string(SHA256 src_sha "${src}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(lf "${WORK_DIR}/${NAME}.lf.cu")
file(WRITE "${lf}" "${src}")

file(REMOVE "${OUTPUT}" "${OUTPUT}.provenance" "${OUTPUT}.srcsha256")

if(MODE STREQUAL "exe")
  execute_process(
    COMMAND "${KATH}" --cpu "${lf}" -o "${OUTPUT}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "kath failed for ${SOURCE} (exit ${rc}):\n${out}\n${err}")
  endif()
  set(prov "exe:kath ${BOOTH_VERSION}")

elseif(MODE STREQUAL "wsl")
  get_filename_component(out_dir "${OUTPUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${out_dir}")
  execute_process(COMMAND wsl.exe -e wslpath -a "${lf}"
    RESULT_VARIABLE rc1 OUTPUT_VARIABLE lin_src OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 120)
  execute_process(COMMAND wsl.exe -e wslpath -a "${out_dir}"
    RESULT_VARIABLE rc2 OUTPUT_VARIABLE lin_dir OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 120)
  if(NOT rc1 EQUAL 0 OR NOT rc2 EQUAL 0)
    message(FATAL_ERROR "wslpath failed; is WSL running?")
  endif()
  get_filename_component(out_name "${OUTPUT}" NAME)
  execute_process(
    COMMAND wsl.exe -e sh -c "${WSL_KATH} --cpu '${lin_src}' -o '${lin_dir}/${out_name}'"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "kath (WSL) failed for ${SOURCE} (exit ${rc}):\n${out}\n${err}")
  endif()
  set(prov "wsl:kath ${BOOTH_VERSION}")

elseif(MODE STREQUAL "prebuilt")
  set(obj "${PREBUILT_DIR}/${NAME}.o")
  foreach(f "${obj}" "${obj}.sha256" "${PREBUILT_DIR}/${NAME}.cu.sha256")
    if(NOT EXISTS "${f}")
      message(FATAL_ERROR "prebuilt mode: missing ${f}")
    endif()
  endforeach()
  file(STRINGS "${PREBUILT_DIR}/${NAME}.cu.sha256" want_src LIMIT_COUNT 1)
  if(NOT want_src STREQUAL src_sha)
    message(FATAL_ERROR "prebuilt mode: ${SOURCE} changed since ${NAME}.o was built "
                        "(source sha256 ${src_sha}, prebuilt was built from ${want_src}). "
                        "Rebuild with kath and run the refresh_prebuilt target.")
  endif()
  file(STRINGS "${obj}.sha256" want_obj LIMIT_COUNT 1)
  file(SHA256 "${obj}" got_obj)
  if(NOT want_obj STREQUAL got_obj)
    message(FATAL_ERROR "prebuilt mode: ${obj} SHA-256 ${got_obj} does not match ${want_obj}")
  endif()
  file(COPY_FILE "${obj}" "${OUTPUT}")
  set(prov "prebuilt:kath ${BOOTH_VERSION}")

else()
  message(FATAL_ERROR "KathCompile: unknown MODE '${MODE}'")
endif()

if(NOT EXISTS "${OUTPUT}")
  message(FATAL_ERROR "kath reported success but ${OUTPUT} was not created")
endif()
file(SIZE "${OUTPUT}" sz)
if(sz EQUAL 0)
  message(FATAL_ERROR "${OUTPUT} is empty")
endif()
file(WRITE "${OUTPUT}.provenance" "${prov}")
file(WRITE "${OUTPUT}.srcsha256" "${src_sha}")
