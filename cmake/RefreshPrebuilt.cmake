# Build-time script (cmake -P): copy a freshly compiled kernel object into
# kernels/prebuilt with its object hash and the hash of the (LF-normalised) source.
# Inputs: OBJ, NAME, DEST.
cmake_minimum_required(VERSION 3.26)

foreach(v OBJ NAME DEST)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "RefreshPrebuilt: ${v} not set")
  endif()
endforeach()
if(NOT EXISTS "${OBJ}.srcsha256")
  message(FATAL_ERROR "RefreshPrebuilt: ${OBJ}.srcsha256 missing; build the kernel first")
endif()
file(READ "${OBJ}.provenance" prov)
if(prov MATCHES "^prebuilt:")
  message(FATAL_ERROR "RefreshPrebuilt: ${NAME} was itself taken from prebuilt; build with kath (exe or wsl mode)")
endif()
file(MAKE_DIRECTORY "${DEST}")
file(COPY_FILE "${OBJ}" "${DEST}/${NAME}.o")
file(SHA256 "${DEST}/${NAME}.o" osha)
file(READ "${OBJ}.srcsha256" ssha)
file(WRITE "${DEST}/${NAME}.o.sha256" "${osha}\n")
file(WRITE "${DEST}/${NAME}.cu.sha256" "${ssha}\n")
message(STATUS "prebuilt ${NAME}.o updated (${osha})")
