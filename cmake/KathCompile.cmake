# Build-time script (cmake -P): run kath for one kernel.
# Inputs: MODE (exe|wsl|prebuilt), SOURCE, OUTPUT, NAME, WORK_DIR, PREBUILT_DIR,
#         KATH (exe mode), WSL_KATH (wsl mode), BOOTH_VERSION,
#         EMIT: cpu (default) -> OUTPUT is the --cpu object;
#               metal         -> OUTPUT is the --metal source and OUTPUT_IR the --ir dump
#                                (input of pcuda_mslgen, the native code paths).
# Writes OUTPUT(.provenance, .srcsha256) [and OUTPUT_IR]. Fails loudly on any problem.
cmake_minimum_required(VERSION 3.26)

foreach(v MODE SOURCE OUTPUT NAME WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "KathCompile: ${v} not set")
  endif()
endforeach()
if(NOT DEFINED EMIT OR EMIT STREQUAL "")
  set(EMIT cpu)
endif()
if(EMIT STREQUAL "metal" AND (NOT DEFINED OUTPUT_IR OR OUTPUT_IR STREQUAL ""))
  message(FATAL_ERROR "KathCompile: EMIT=metal needs OUTPUT_IR")
endif()

# Normalise line endings so CRLF checkouts produce identical input (and source hashes).
file(READ "${SOURCE}" src)
string(REPLACE "\r\n" "\n" src "${src}")
string(SHA256 src_sha "${src}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(lf "${WORK_DIR}/${NAME}.${EMIT}.lf.cu")
file(WRITE "${lf}" "${src}")

file(REMOVE "${OUTPUT}" "${OUTPUT}.provenance" "${OUTPUT}.srcsha256")
if(EMIT STREQUAL "metal")
  file(REMOVE "${OUTPUT_IR}")
endif()

# Prebuilt artifacts per EMIT kind: <name>.o, or <name>.metal + <name>.ir, each with .sha256.
if(EMIT STREQUAL "cpu")
  set(prebuilt_pairs "${NAME}.o=${OUTPUT}")
elseif(EMIT STREQUAL "metal")
  set(prebuilt_pairs "${NAME}.metal=${OUTPUT}" "${NAME}.ir=${OUTPUT_IR}")
else()
  message(FATAL_ERROR "KathCompile: unknown EMIT '${EMIT}'")
endif()

# Runs kath with one flag and writes `dest`. --ir prints to stdout; the others take -o.
function(run_kath flag dest)
  if(MODE STREQUAL "exe")
    if(flag STREQUAL "--ir")
      execute_process(COMMAND "${KATH}" --ir "${lf}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
      if(rc EQUAL 0)
        file(WRITE "${dest}" "${out}")
      endif()
    else()
      execute_process(COMMAND "${KATH}" ${flag} "${lf}" -o "${dest}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
    endif()
  else()  # wsl
    get_filename_component(dir "${dest}" DIRECTORY)
    get_filename_component(name "${dest}" NAME)
    file(MAKE_DIRECTORY "${dir}")
    execute_process(COMMAND wsl.exe -e wslpath -a "${lf}"
      RESULT_VARIABLE rc1 OUTPUT_VARIABLE lin_src OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 120)
    execute_process(COMMAND wsl.exe -e wslpath -a "${dir}"
      RESULT_VARIABLE rc2 OUTPUT_VARIABLE lin_dir OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 120)
    if(NOT rc1 EQUAL 0 OR NOT rc2 EQUAL 0)
      message(FATAL_ERROR "wslpath failed; is WSL running?")
    endif()
    if(flag STREQUAL "--ir")
      execute_process(COMMAND wsl.exe -e sh -c "${WSL_KATH} --ir '${lin_src}'"
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
      if(rc EQUAL 0)
        file(WRITE "${dest}" "${out}")
      endif()
    else()
      execute_process(COMMAND wsl.exe -e sh -c "${WSL_KATH} ${flag} '${lin_src}' -o '${lin_dir}/${name}'"
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
    endif()
  endif()
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "kath ${flag} failed for ${SOURCE} (exit ${rc}):\n${out}\n${err}")
  endif()
  if(NOT EXISTS "${dest}")
    message(FATAL_ERROR "kath ${flag} reported success but ${dest} was not created")
  endif()
endfunction()

if(MODE STREQUAL "exe" OR MODE STREQUAL "wsl")
  if(EMIT STREQUAL "cpu")
    run_kath(--cpu "${OUTPUT}")
  else()
    run_kath(--metal "${OUTPUT}")
    run_kath(--ir "${OUTPUT_IR}")
  endif()
  set(prov "${MODE}:kath ${BOOTH_VERSION}")

elseif(MODE STREQUAL "prebuilt")
  set(src_hash_file "${PREBUILT_DIR}/${NAME}.cu.sha256")
  if(NOT EXISTS "${src_hash_file}")
    message(FATAL_ERROR "prebuilt mode: missing ${src_hash_file}")
  endif()
  file(STRINGS "${src_hash_file}" want_src LIMIT_COUNT 1)
  if(NOT want_src STREQUAL src_sha)
    message(FATAL_ERROR "prebuilt mode: ${SOURCE} changed since the ${NAME} artifacts were built "
                        "(source sha256 ${src_sha}, prebuilt was built from ${want_src}). "
                        "Rebuild with kath and run the refresh_prebuilt target.")
  endif()
  foreach(pair IN LISTS prebuilt_pairs)
    string(REPLACE "=" ";" pair "${pair}")
    list(GET pair 0 art)
    list(GET pair 1 dest)
    set(f "${PREBUILT_DIR}/${art}")
    if(NOT EXISTS "${f}" OR NOT EXISTS "${f}.sha256")
      message(FATAL_ERROR "prebuilt mode: missing ${f} or ${f}.sha256")
    endif()
    file(STRINGS "${f}.sha256" want LIMIT_COUNT 1)
    file(SHA256 "${f}" got)
    if(NOT want STREQUAL got)
      message(FATAL_ERROR "prebuilt mode: ${f} SHA-256 ${got} does not match ${want}")
    endif()
    file(COPY_FILE "${f}" "${dest}")
  endforeach()
  set(prov "prebuilt:kath ${BOOTH_VERSION}")

else()
  message(FATAL_ERROR "KathCompile: unknown MODE '${MODE}'")
endif()

file(SIZE "${OUTPUT}" sz)
if(sz EQUAL 0)
  message(FATAL_ERROR "${OUTPUT} is empty")
endif()
file(WRITE "${OUTPUT}.provenance" "${prov}")
file(WRITE "${OUTPUT}.srcsha256" "${src_sha}")
