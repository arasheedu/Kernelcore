# CUDA CPU Runtime POC: Booth CPU Backend + PantheonSim-Inspired Runtime

**Status:** Implementation plan  
**Target:** Small, testable proof of concept (not a full CUDA replacement)  
**Primary language:** C++20 host runtime, CUDA C kernel source  
**Initial platform:** Ubuntu 24.04 on Linux or WSL2  
**Compiler backend:** Booth `kath --cpu`  
**Runtime design reference:** PantheonSim / VirtualGPU architecture

---

## 1. Objective

Build a small runtime that accepts a deliberately limited subset of CUDA-style application code, compiles CUDA kernel source to native x86-64 object code using Booth, and executes the resulting kernel through a PantheonSim-inspired host runtime.

The POC must demonstrate:

1. CUDA `.cu` kernel source can be compiled by Booth's CPU backend without an NVIDIA GPU.
2. A host application can discover a CPU device, allocate device-style memory, copy data, launch a registered kernel, synchronize, and retrieve results.
3. The runtime has a backend boundary that could later host a native NVIDIA CUDA backend.
4. An optional chunked execution experiment can distribute independent 1-D work over multiple CPU workers, while clearly documenting that this is a host-side decomposition and not full CUDA grid/block emulation.

**Success means functional proof and clean architecture—not GPU-equivalent performance or full CUDA compatibility.**

## 2. Important constraints discovered in the upstream projects

### Booth

Booth's `--cpu` backend compiles CUDA source to a host-runnable x86-64 object. Its documented calling convention is the kernel's declared parameters followed by an additional `nthreads` argument. For a 1-D kernel, the CPU backend runs the kernel body over logical thread IDs from zero to `nthreads - 1`.

Current documented limitations relevant to this POC:

- CPU backend generates device/kernel code only; it does not generate the host application.
- CPU code generation is correctness-first and currently has no register allocator.
- It executes one block per call.
- `tl.load` masks are not honored; the documented safe pattern is to set `nthreads` to the exact element count.
- The CPU backend is not yet a high-performance N-core CUDA scheduler.

Booth's documented CPU object format is System V ELF. Link and run the generated object on Linux or WSL, not native MinGW. Although Booth distributes Windows compiler binaries, the generated CPU object is not a native Windows COFF object.

### PantheonSim

PantheonSim provides useful runtime design concepts: device discovery, a runtime/driver facade, allocation tracking, memory operations, streams, events, error codes, launch validation, diagnostics, and a clear separation between vendor frontends and a common execution engine.

Do **not** make the POC depend on PantheonSim's internal code or pretend to implement its CUDA driver ABI. Use it as an architectural reference. PantheonSim executes compiled CUDA programs by interpreting PTX/SASS on a simulated GPU; this POC instead uses Booth-generated native CPU object code.

### Consequence for scope

The first POC is a **CUDA-inspired source/runtime subset**, not a drop-in `libcudart` replacement. Use project-prefixed APIs such as `kcMalloc` and `kcLaunchKernel` to avoid collisions with NVIDIA's actual CUDA headers and libraries.

## 3. Non-goals for POC 0.1

Do not attempt these in the first milestone:

- Full CUDA Runtime API or Driver API ABI compatibility.
- Running arbitrary precompiled CUDA binaries, PTX, cubins, or fatbins.
- Parsing or transforming CUDA launch syntax (`kernel<<<grid, block>>>`) into the host runtime automatically.
- Full multi-dimensional grid/block semantics.
- General `__shared__` memory, `__syncthreads`, warp intrinsics, cooperative groups, or device-side launches.
- Transparent support for every CUDA library.
- GPU performance simulation.
- Native Windows execution of Booth-generated CPU objects.
- A production-quality JIT compiler or dynamic loader for arbitrary generated kernel objects.

The POC host program will call a typed kernel wrapper. Kernel compilation happens at build time and the generated object is linked into the executable.

## 4. Proposed architecture

```text
             CUDA C kernel source (.cu)
                        |
                        v
               Booth compiler (kath)
                    --cpu
                        |
                        v
              x86-64 relocatable object
                        |
              link into POC executable
                        |
                        v
+--------------------------------------------------+
|             POC Host Application                 |
|                                                  |
|  typed kernel adapter / KernelRegistry           |
|  POC runtime API (Kernelcore*)                        |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|                 Runtime Core                     |
|                                                  |
| DeviceManager | AllocationRegistry | Error model |
| Stream/Event model | Launch validation           |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
|                  Backend API                     |
+--------------------------+-----------------------+
                           |
                 +---------+---------+
                 |                   |
                 v                   v
          CpuBackend            NvidiaBackend
          (POC 0.1)             (future; stub only)
                 |
                 v
       Booth-generated native kernel
       + optional host-side chunk pool
```

### Design rules

1. Runtime core must not include Booth compiler internals.
2. Runtime core must not include NVIDIA headers.
3. Generated kernel ABI details must be isolated in typed adapters.
4. Memory ownership must be tracked by the runtime, not inferred from pointer values alone.
5. Unsupported operations must return explicit errors; never silently pretend they worked.
6. CPU and future GPU backends must implement the same high-level backend contract where semantics genuinely match.

## 5. Repository layout

```text
cuda-cpu-poc/
├── CMakeLists.txt
├── README.md
├── cmake/
│   └── BoothKernel.cmake
├── kernels/
│   └── vector_add.cu
├── include/kernelcore/
│   ├── api.hpp
│   ├── types.hpp
│   ├── backend.hpp
│   ├── kernel_registry.hpp
│   └── error.hpp
├── src/
│   ├── runtime.cpp
│   ├── device_manager.cpp
│   ├── allocation_registry.cpp
│   ├── cpu_backend.cpp
│   ├── stream.cpp
│   ├── event.cpp
│   └── kernel_registry.cpp
├── adapters/
│   └── vector_add_adapter.cpp
├── app/
│   └── main.cpp
├── tests/
│   ├── test_memory.cpp
│   ├── test_vector_add.cpp
│   ├── test_stream.cpp
│   └── test_chunked_vector_add.cpp
└── scripts/
    ├── bootstrap_wsl.sh
    └── run_poc.sh
```

## 6. Runtime API for POC 0.1

Use a project-specific namespace and names. The API should look familiar to CUDA developers without claiming binary/source compatibility with CUDA Runtime API.

```cpp
namespace kc {

enum class Error {
    Success,
    InvalidValue,
    InvalidDevice,
    OutOfMemory,
    InvalidPointer,
    UnsupportedOperation,
    LaunchFailure,
    NotReady
};

enum class CopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost
};

struct DeviceProperties {
    char name[128];
    unsigned worker_count;
    size_t total_memory;
    bool is_hardware_gpu;
};

Error getDeviceCount(int* count);
Error getDeviceProperties(DeviceProperties* props, int device);
Error setDevice(int device);
Error malloc(void** ptr, size_t bytes);
Error free(void* ptr);
Error memcpy(void* dst, const void* src, size_t bytes, CopyKind kind);
Error deviceSynchronize();

}
```

### Runtime semantics

- POC exposes one logical CPU device (`device 0`).
- `malloc` uses aligned host allocation and registers the allocation.
- `free` validates that the pointer is a live registered allocation.
- `memcpy` validates pointer ranges for device-side operands and performs a host copy.
- `deviceSynchronize` waits for all submitted work in the default stream.
- Return stable error values and include diagnostic text in a separate last-error facility or result object.

Do not report the CPU device as an NVIDIA GPU. Device properties must identify it as a CPU execution backend.

## 7. Kernel ABI integration with Booth

### First kernel

Start with the smallest supported elementwise CUDA kernel:

```cpp
extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* out,
    int count)
{
    const int i = threadIdx.x;
    if (i < count)
        out[i] = a[i] + b[i];
}
```

Compile it using Booth:

```bash
kath --cpu kernels/vector_add.cu -o build/vector_add.o
```

The generated host-callable function has the kernel's declared arguments plus Booth's trailing `nthreads` parameter. Confirm the exact exported symbol and signature from the generated object and compiler diagnostics before writing the adapter. Do not assume name mangling or ABI details without checking them.

### Typed adapter

Keep the Booth-specific ABI in one translation unit:

```cpp
extern "C" void vector_add(
    const float* a,
    const float* b,
    float* out,
    int count,
    int nthreads);

namespace kc::kernels {
void launchVectorAddCpu(
    const float* a,
    const float* b,
    float* out,
    int count)
{
    vector_add(a, b, out, count, count);
}
}
```

The signature above is the expected POC adapter shape; adjust it to the exact ABI Booth emits for the selected source and pinned compiler version.

### Kernel registry

For POC 0.1, use explicit typed registration rather than a generic `void*` argument marshaller. A generic kernel ABI requires metadata for parameter layout, alignment, pointer fixups, and compiler-generated launch wrappers; that is beyond this proof of concept.

```cpp
struct KernelEntry {
    const char* name;
    void (*launch_cpu)(const void* typed_args);
};
```

A later compiler integration can generate these adapters and metadata automatically.

## 8. Backend interface

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* name() const noexcept = 0;
    virtual kc::Error allocate(void** ptr, size_t bytes) = 0;
    virtual kc::Error release(void* ptr) = 0;
    virtual kc::Error copy(
        void* dst, const void* src, size_t bytes,
        kc::CopyKind kind) = 0;
    virtual kc::Error synchronize() = 0;
};
```

Implement `CpuBackend` now. Add `NvidiaBackend` only as a compile-time optional stub or future milestone. Do not link CUDA libraries in the CPU-only build.

The kernel-launch path should initially be a typed CPU adapter call. The interface can later gain a backend-specific launch operation once both CPU and GPU kernel descriptors exist.

## 9. Allocation registry

Track every runtime-owned allocation:

```cpp
struct AllocationInfo {
    void* base;
    size_t bytes;
    int device_id;
    bool live;
};
```

Required behavior:

- Register after successful allocation.
- Reject double free and unknown pointers.
- Validate that a copy's requested range fits inside a registered allocation when the direction requires a device pointer.
- Detect interior pointers and compute their offset safely.
- Protect registry operations with a mutex.
- Keep allocation tracking separate from the allocator implementation.

For the POC, use `std::aligned_alloc` or an equivalent platform wrapper with correct size alignment, and pair it with the matching deallocation function.

## 10. Streams and events

Implement a minimal stream abstraction inspired by PantheonSim's runtime surface.

### POC 0.1

- One default stream.
- Synchronous kernel launch.
- `deviceSynchronize()` is effectively a completion fence.

### POC 0.2

- `Stream` owns a FIFO task queue and one worker thread.
- Operations submitted to one stream execute in submission order.
- `streamSynchronize()` waits for that stream's queue to drain.
- `Event` records a completion marker in a stream.
- `eventSynchronize()` waits for the marker.
- Cross-stream waits can be deferred until the basic ordering tests pass.

Do not advertise `memcpyAsync` as asynchronous until host-buffer lifetime and operation ordering are implemented and tested.

## 11. N-core execution experiment

This is a separate experiment, not an assumption about Booth's current CPU backend.

Booth documents its CPU backend as one block per call, with the kernel body lowered into a logical thread loop. It does not currently promise a general multi-block CPU scheduler or a high-performance worker-pool implementation.

For an independent 1-D elementwise kernel, the host runtime can experiment with chunking:

1. Divide the input range into `worker_count` disjoint chunks.
2. Submit one host task per chunk to a persistent worker pool.
3. Each task invokes the Booth-generated kernel with pointers offset to the chunk start and `nthreads = chunk_length`.
4. Wait for all tasks and propagate the first error.

This requires a kernel contract in which each invocation's `threadIdx.x` is relative to its chunk and the kernel has no cross-block communication. It is **not** a general implementation of CUDA grid semantics. Do not use this strategy for reductions, shared-memory kernels, barriers, global atomics, or kernels whose result depends on block IDs.

The chunked experiment should use a second explicit kernel/adapter if the simple Booth lowering cannot support the pointer-offset contract cleanly. Do not silently rewrite arbitrary kernels.

## 12. Implementation phases and tasks

### Phase 0 — Upstream spike and pinning (1–2 days)

- Clone Booth and inspect its current release/tag and license.
- Pin an exact Booth tag or commit in the POC README and build scripts.
- Build or download `kath` for the WSL/Linux environment.
- Run `kath --version` and `kath doctor` if available in the pinned version.
- Compile Booth's own `examples/cpu_launch_vadd.c` path and run it.
- Compile the minimal CUDA `vector_add.cu` with `--cpu`.
- Inspect the resulting object with `nm` and `readelf` to confirm symbol name, object format, and function ABI.
- Record all compiler limitations and exact command lines.

**Exit criteria:** A Booth-generated object is linked into a normal C/C++ host executable and produces correct vector-add output on a CPU-only Linux/WSL machine.

### Phase 1 — Runtime skeleton (2–3 days)

- Create CMake project using C++20.
- Add `IBackend`, `CpuBackend`, `DeviceManager`, `Error`, and `DeviceProperties`.
- Implement one CPU logical device.
- Implement aligned allocation and basic copy operations.
- Add allocation registry and pointer validation.
- Add unit tests for allocation, copy, invalid pointer, double free, and bounds.

**Exit criteria:** Runtime memory API tests pass without Booth integration.

### Phase 2 — Booth build integration (1–2 days)

- Add a CMake custom command that invokes `kath --cpu` on `kernels/vector_add.cu`.
- Link the generated object into the host executable.
- Add a typed adapter for the exact emitted symbol signature.
- Make the build fail if Booth returns an error or does not produce the expected object.
- Record the Booth version in build output.

**Exit criteria:** A clean CMake build compiles the `.cu` file with Booth and links it without CUDA Toolkit or NVIDIA driver dependencies.

### Phase 3 — End-to-end runtime launch (2–3 days)

- Implement `kcLaunchVectorAdd` using the typed adapter.
- Validate device selection and pointer ownership before launch.
- Track launch errors and synchronization state.
- Add an end-to-end test: allocate → H2D copy → launch → synchronize → D2H copy → compare.
- Add deterministic test inputs and boundary sizes (1, 7, 256, 257, and a larger buffer).

**Exit criteria:** Host application uses only the POC runtime API and returns correct results.

### Phase 4 — Streams/events (2–4 days)

- Implement FIFO stream queue.
- Add stream synchronization.
- Add event record and wait.
- Test ordering with host-side marker tasks and kernel launches.
- Ensure exceptions/errors are captured and returned to the synchronizing caller.

**Exit criteria:** Two streams execute independently where safe, while each stream preserves its own submission order.

### Phase 5 — N-core chunked experiment (3–5 days)

- Add persistent worker pool with configurable worker count.
- Add chunk planner for independent 1-D kernels only.
- Add chunked vector-add adapter using pointer offsets and exact per-chunk `nthreads`.
- Compare one-worker and N-worker output against the scalar reference.
- Benchmark scheduler overhead and throughput for several input sizes.
- Document that this is host-side partitioning, not generic CUDA block scheduling.

**Exit criteria:** Correct disjoint chunk execution on 1, 2, 4, and available-core worker counts; no data races under ThreadSanitizer where supported.

### Phase 6 — Optional GPU parity (later)

- Add optional NVIDIA backend behind a CMake option.
- Compile the same CUDA kernel with `nvcc` for the real GPU.
- Use a separate typed GPU adapter.
- Run identical test vectors on CPU and GPU and compare within documented floating-point tolerances.
- Keep CPU-only builds working when CUDA Toolkit is absent.

This phase is not required to prove the CPU runtime concept.

## 13. Build and run workflow

Initial development should happen in Ubuntu under WSL2 or native Linux. Booth's CPU object is documented as System V ELF and is not directly linkable by a native MinGW Windows executable.

Example high-level workflow:

```bash
# 1. Build or install Booth and ensure kath is on PATH.
kath --version

# 2. Compile the CUDA kernel to a CPU object.
kath --cpu kernels/vector_add.cu -o build/vector_add.o

# 3. Configure and build the C++ host runtime.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 4. Run the POC.
./build/kc_poc

# 5. Run tests.
ctest --test-dir build --output-on-failure
```

CMake should use a custom command for Booth, with `KATH_EXECUTABLE` configurable:

```bash
cmake -S . -B build -DKATH_EXECUTABLE=/path/to/kath
```

Do not assume Booth's CMake package is installed or stable across 0.x versions. A direct custom command is sufficient for this small POC; integration through `find_package(Booth)` can be evaluated after pinning a compatible release.

## 14. Test matrix and acceptance criteria

| Area | Test | Pass condition |
|---|---|---|
| Compiler | Compile vector-add `.cu` with `kath --cpu` | Object emitted, no GPU required |
| ABI | Inspect/link generated symbol | Typed adapter calls correct signature |
| Device | Enumerate devices | One CPU device identified accurately |
| Memory | Allocate/free/copy | Correct bytes; invalid operations rejected |
| Kernel | Vector add | Every output matches reference |
| Bounds | Sizes 1, 7, 256, 257 | No out-of-range writes |
| Stream | Ordered operations | Submission order preserved |
| Event | Record/synchronize | Completion observed correctly |
| Chunking | 1/2/4/N workers | Same result as serial reference |
| Build isolation | Build without CUDA Toolkit | CPU POC still builds and runs |
| Diagnostics | Unsupported operation | Explicit error, never silent success |

### Definition of done

POC 0.1 is complete when:

- A clean Linux/WSL2 build produces the executable from the repository.
- Booth is pinned to an exact version/commit.
- No NVIDIA GPU, CUDA Toolkit, or NVIDIA driver is required for the CPU path.
- The host application uses the POC runtime for device selection, allocation, copying, launch, and synchronization.
- Vector-add tests pass across boundary sizes.
- The runtime reports unsupported features clearly.
- The architecture leaves a clean seam for a future NVIDIA backend.

POC 0.2 is complete when streams/events work and the chunked 1-D worker-pool experiment passes correctness and race checks.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Booth CPU ABI changes | Pin exact commit/tag; isolate ABI in one adapter; add symbol/ABI smoke test |
| Booth backend limitations | Keep kernel subset small; fail build/tests when unsupported constructs appear |
| Confusing CPU execution with GPU emulation | Name device/backend honestly; document semantic differences |
| Incorrect generic kernel launch assumptions | Use typed adapters for POC; do not invent a universal ABI |
| Chunking breaks CUDA semantics | Restrict to independent 1-D elementwise kernels and use explicit opt-in |
| Windows object-format mismatch | Use WSL2/Linux for initial build; treat native Windows port as separate work |
| Accidental CUDA dependency | Keep CUDA headers/libraries out of CPU target and CI build |

## 16. Upstream references

- Booth repository: https://github.com/Zaneham/Booth
- Booth usage guide: https://github.com/Zaneham/Booth/blob/master/docs/usage.md
- Booth feature status: https://github.com/Zaneham/Booth/blob/master/docs/features.md
- Booth CPU vector-add example: https://github.com/Zaneham/Booth/blob/master/examples/cpu_launch_vadd.c
- Booth CPU matmul example: https://github.com/Zaneham/Booth/blob/master/examples/cpu_launch_matmul.c
- Booth CMake consumer guide: https://github.com/Zaneham/Booth/blob/master/docs/cmake.md
- PantheonSim repository: https://github.com/pantheongpu/pantheonsim
- PantheonSim architecture: https://github.com/pantheongpu/pantheonsim/blob/main/ARCHITECTURE.md
- PantheonSim getting-started docs: https://pantheonsim.com/docs/

## 17. Recommended first implementation decision

Start with **one CUDA vector-add kernel, one Booth-generated CPU object, one typed host adapter, and a small synchronous runtime**. Do not start by implementing a generic CUDA API, a generic kernel argument ABI, or full CUDA scheduling.

Once the end-to-end path is proven, add streams/events, then the explicitly limited N-core chunking experiment. This sequence tests the central integration assumption quickly and keeps the POC small enough to feed into an agentic coding workflow.
