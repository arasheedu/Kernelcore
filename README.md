# pcuda: CUDA kernels on Windows x64 CPU cores (POC)

pcuda runs CUDA C kernels on the CPU of a normal Windows PC, with no NVIDIA GPU, CUDA Toolkit or driver.
[Booth](https://github.com/Zaneham/Booth) (`kath --cpu`) compiles each kernel to x86-64 code at build
time. The pcuda runtime loads that code and runs it behind a small CUDA-style API (`pcudaMalloc`,
`pcudaMemcpy`, `pcudaLaunchKernel`, streams, events).

It is a proof of concept, not a CUDA replacement. See `cuda_cpu_poc_plan.md` for the design and
`cuda_cpu_poc_plan_review.md` for why the design is Windows-native.

## Quick start

```powershell
.\scripts\bootstrap.ps1 -FetchKath   # check toolchain; download pinned kath.exe (SHA-256 verified)
.\scripts\build.ps1                  # configure + build + run all tests (preset windows-msvc)
.\build\windows-msvc\pcuda_poc.exe   # demo: 1M-element vector add, prints PASS
.\build\windows-msvc\pcuda_bench.exe # serial vs multi-core chunked timings
```

Requirements: Windows 10/11 x64, Visual Studio 2022 with "Desktop development with C++" (MSVC, ml64),
CMake ≥ 3.26 and Ninja. Ninja is bundled with Visual Studio. `build.ps1` loads the Visual Studio
environment itself, so any PowerShell window works.

Presets: `windows-msvc` (default), `windows-msvc-debug`, `windows-msvc-asan` (AddressSanitizer),
`windows-msvc-prebuilt` (no `kath` needed) and `windows-clangcl` (untested here; clang-cl wasn't installed).

## How it works

```text
kernels/vector_add.cu --kath --cpu--> ELF64 object (System V ABI)
    --pcuda_objcheck--> embedded byte array in the .exe
    --runtime: validate, map (W^X), relocate--> called through a MASM Win64->System V thunk
```

Booth emits Linux-style objects: ELF format and the System V calling convention. Windows linkers
can't use them, so pcuda handles them itself:

- **Loader** (`src/cpu/booth_elf.cpp`, `booth_image.cpp`): a strict, bounds-checked parser. It accepts
  only the six-section, single-relocation-type subset Booth emits and rejects anything else. The same
  parser runs at build time as `pcuda_objcheck`.
- **Thunk** (`src/cpu/sysv_invoke.asm`): saves every Win64 non-volatile register, including `rdi`,
  `rsi` and `xmm6-15` (which Booth code clobbers), plus MXCSR and the x87 control word. It passes every
  integer as 64 bits; Booth's hidden `nthreads` argument is read as 64 bits.
- **Math shims** (`src/cpu/sysv_shims.asm`): `sinf`, `cosf`, `exp2f` and `log2f` calls from kernels get
  the 32-byte shadow space that Windows functions are allowed to overwrite.
- **Fault containment** (`src/cpu/fault_guard.cpp`): an access violation, divide-by-zero or stack
  overflow inside a kernel becomes `Error::KernelFault`. After that, other calls return
  `ContextPoisoned` until `pcudaDeviceReset()`, as with CUDA's sticky errors. The process keeps running.
- **Guard pages**: every allocation ends against a no-access page, so an overrun faults at the first
  byte past the end.

## Getting kernel objects (`-DPCUDA_KATH_MODE=`)

| Mode | Source | When |
|---|---|---|
| `exe` | pinned `kath.exe` (SHA-256 checked) | normal development |
| `wsl` | Linux `kath` inside WSL (`wsl.exe -e sh scripts/install_kath_wsl.sh`) | no Windows binary |
| `prebuilt` | `kernels/prebuilt/*.o`, tied to the kernel source's hash | no `kath` at all (CI, other machines) |
| `auto` (default) | first that works: exe → wsl → prebuilt | |

After changing a kernel, run `cmake --build --preset windows-msvc --target refresh_prebuilt` to update the
prebuilt objects. A prebuilt object whose source has changed is refused at build time. The Windows and
WSL builds of `kath` produce byte-identical objects.

## Using the API

```cpp
#include <pcuda/api.hpp>
#include <pcuda/kernels.hpp>

pcudaInit();
float *a, *b, *out;
pcudaMalloc((void**)&a, n * sizeof(float));   // also b, out
pcudaMemcpy(a, host_a, n * sizeof(float), pcuda::CopyKind::HostToDevice);
pcuda::Error e = pcuda::kernels::launchVectorAdd(a, b, out, n);   // synchronous on the default stream
if (e != pcuda::Error::Success) printf("%s: %s\n", pcudaGetErrorString(e), pcudaGetLastErrorDetail());
pcudaMemcpy(host_out, out, n * sizeof(float), pcuda::CopyKind::DeviceToHost);
pcudaShutdown();
```

To add a kernel:
1. Write `kernels/<name>.cu` with exactly one `__global__` function.
2. Add `booth_add_kernel(TARGET pcuda NAME <name> SOURCE kernels/<name>.cu)`.
3. Write a typed adapter (see `adapters/vector_add_adapter.cpp`) and list it in `adapters/builtin_kernels.cpp`,
   or register it at runtime with `pcudaRegisterKernel`.

## Kernel contract (Booth 0.6.1 CPU backend)

- One `__global__` per `.cu` file. Booth emits only the first one.
- At most 5 integer/pointer parameters plus Booth's hidden `nthreads`, and at most 8 `float` parameters.
  Enforced when arguments are marshalled.
- Each call runs one block: `threadIdx.x` goes from 0 to `nthreads-1`, `blockIdx.x` is 0 and `blockDim.x` is `nthreads`.
  Write the index as `blockIdx.x * blockDim.x + threadIdx.x`: it reduces to `threadIdx.x` here and is the
  correct global index on a GPU, so the same file also builds with `nvcc`.
- No `__shared__`, `__syncthreads`, atomics or warp intrinsics.
- `sinf`/`cosf` arguments are converted to turns and back in float before the call, so results can differ
  from a direct CRT call by a few ULP.
- Chunked multi-core launches (`pcudaLaunchKernelChunked`) are opt-in per kernel and only correct for
  independent 1-D element-wise kernels. They split the work on the host; this isn't CUDA grid scheduling.
  Chunking pays off from about 256K elements; worker wake-up costs about 40 µs per launch.

## Side by side with real CUDA (optional)

On a machine with an NVIDIA GPU and the CUDA Toolkit, `pcuda_vs_cuda` runs the same `.cu` files on
pcuda (compiled by Booth) and on the GPU (compiled by `nvcc`). It checks the results against each other
and against a double-precision reference, then times both.

```powershell
.\scripts\build.ps1 -Preset windows-msvc-nvidia          # also runs pcuda_vs_cuda --check under ctest
.\build\windows-msvc-nvidia\pcuda_vs_cuda.exe            # full table, 1K..16M elements (about 1 minute)
.\build\windows-msvc-nvidia\pcuda_vs_cuda.exe --quick    # up to 1M elements
```

Columns: plain C++ at `/O2`, pcuda on 1 thread and on all cores, pcuda including `pcudaMemcpy` in/out,
CUDA kernel time (`cudaEvent`), and CUDA including copies over PCIe (pinned host memory).
The CPU-only build never needs CUDA; this target exists only with `-DPCUDA_ENABLE_NVIDIA=ON`.

CUDA 12.3 and older reject MSVC 19.40+ (VS 2022 17.10 and later). The build then adds
`-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` and prints a warning.
CUDA 12.4 or newer doesn't need it.

Results on an i5-13400F (16 logical processors) and RTX 4060, 16M elements:

| Kernel | C++ /O2 | pcuda, 16 threads | CUDA kernel | pcuda + copies | CUDA + copies |
|---|---|---|---|---|---|
| `vector_add` | 6.2 ms | 8.7 ms | 0.80 ms | 18.1 ms | 16.3 ms |
| `scale_add` | 5.2 ms | 8.4 ms | 0.89 ms | 16.4 ms | 10.9 ms |
| `trig_mix` | 170 ms | 62 ms | 0.54 ms | 67 ms | 10.9 ms |

- Kernel only, the GPU is 10–54× faster for the memory-bound kernels and 115–170× for `trig_mix`.
- With copies included, memory-bound kernels are roughly even (1.0–1.5×), because PCIe transfer
  dominates. Below about 256K elements pcuda wins outright: a CUDA launch-and-wait plus copies costs
  50–80 µs.
- `vector_add` results are bit-identical. `scale_add` differs by at most 1 ULP, because `nvcc` fuses
  `x*s+b` into one FMA and Booth rounds twice. `trig_mix` differs by up to 16 ULP (different `sinf`
  etc. implementations); both sides stay within 1e-6 relative error of double precision.
- Booth's code on one thread is about 10× slower than the C++ loop for simple kernels: it keeps every
  value on the stack and doesn't vectorize.

## Runtime settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PCUDA_GUARD_PAGES` | `1` | `0` uses `_aligned_malloc` instead of guard-page allocations |
| `PCUDA_KERNEL_STACK_MB` | `8` | Stack reserve for stream and worker threads (1–1024) |
| `PCUDA_ALLOW_EMULATION` | unset | `1` lets the runtime start under x64 emulation on ARM64 (unsupported) |

## Limits and known issues

- Executables are linked with `/CETCOMPAT:NO`. Fault recovery rewrites RSP/RIP, which hardware shadow
  stacks would reject.
- Processes under an Arbitrary Code Guard policy get `Error::DynamicCodeBlocked`.
- Host funcs (`pcudaLaunchHostFunc`) must not call pcuda APIs. Doing so returns `UnsupportedOperation`.
- `pcudaStreamWaitEvent` (cross-stream waits) is not implemented.
- ThreadSanitizer isn't available on Windows. Chunk disjointness is checked by the runtime on every
  chunked launch and by tests.
