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

## Code paths: booth, native and simd

Booth's front end and optimizer produce good SSA IR (`kath --ir`), but its x86 back end has no register
allocator: every value goes through the stack. The same IR also comes out of Booth's Metal back end as
plain structured C. So each kernel is built three ways from the one `.cu` file:

| Path | Built from | Runs as |
|---|---|---|
| `booth` | `kath --cpu` object | Booth's x86 code, through the System V thunk |
| `native` | `kath --metal` + `kath --ir` → `pcuda_mslgen` → MSVC `/O2` | one CUDA thread at a time |
| `simd` | the same, translated to AVX2 | 8 CUDA threads per 256-bit register, like a GPU warp |

- `tools/mslgen.cpp` reads the Metal text for structure and the IR for exact semantics. The Metal
  output loses byte signedness and prints `sin`/`cos` in turns without converting back. The generator
  fixes both from the IR, and fails the build on an IR operation it can't reproduce exactly.
- `simd` keeps values that are the same in every lane (loop counters, `x[j]` loads) scalar. It
  masks divergent `if`s and loops, and touches memory only in active lanes. A kernel it can't translate
  (for example one calling an unknown helper function) uses `native`.
- `simd` also tracks values of the form "lane-0 value + constant × lane" (thread indices and the
  addresses built from them) as scalars. A group of 8 threads starts at a multiple of 8, so the
  generator can prove `row / 8` is the same in every lane and `row % 8` is the lane number. Loads
  and stores through such addresses become single vector loads with no run-time checks. A kernel can
  promise that a run-time value is a non-negative multiple of 8 with a `pcuda_aligned8(x)` helper
  (the identity; see `kernels/llm/attn_scores.cu`). Divisions by it then stay uniform too, and the
  simd path checks the promise and faults if it is broken.
- All three paths are bit-identical for race-free kernels (`test_codegen`, and every kernel test runs
  on each path). Native and simd code runs under the same thunk and fault guard, so out-of-bounds
  accesses, divide-by-zero and stack overflow are contained the same way.
- Choose with `pcudaSetKernelCodegen(pcuda::Codegen::...)` or `PCUDA_CODEGEN=auto|booth|native|simd`.
  `auto` (default) picks simd, then native, then booth.

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

Graphs work like CUDA graphs. Launches between `pcudaStreamBeginCapture` and
`pcudaStreamEndCapture` are validated and recorded, not run. `pcudaGraphLaunch` then runs them in
order as one job on the worker pool: each kernel's chunks run in parallel, and the next kernel starts
at a barrier. There is no stream hand-off or thread wake-up per kernel. Results match running the
launches one by one.

```cpp
pcudaStreamBeginCapture(nullptr);            // the default stream
pcudaLaunchKernelChunked("vector_add", &args1, workers);
pcudaLaunchKernel("scale_add", &args2);      // reads vector_add's output
pcudaGraph_t g;
pcudaStreamEndCapture(nullptr, &g);
pcudaGraphLaunch(g);                         // synchronous; or pass a stream
pcudaGraphDestroy(g);
```

During a capture, copies, memsets, frees and synchronization return `UnsupportedOperation`, as in
CUDA, because they would run ahead of the recorded work. After a failing kernel the rest of the
graph is skipped, and the error detail names the graph node.

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

Results on an i5-13400F (16 logical processors) and RTX 4060, 16M elements, measured on the `booth`
code path before the native and simd paths existed (`PCUDA_CODEGEN=booth` reproduces them):

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

### LLM decode vs Ollama (`pcuda_llm_bench`, `scripts\compare_llm.ps1`)

`kernels/llm/` holds the kernels of a Llama decode step: 4-bit and 8-bit matrix-vector, RMS norm,
RoPE, attention scores, softmax, attention values and SwiGLU. They use the subset Booth supports: one
thread per output, no shared memory, no warp intrinsics. `pcuda_llm_bench` builds a model with
Llama 3.2 shapes and random weights and generates tokens on every pcuda path, on CUDA and in plain C++.
`compare_llm.ps1` runs Ollama's `llama3.2` first and matches it:

- the same shapes (Llama 3.2 3B);
- the same weight bytes per token: Q4 2.01 GB here, against Ollama's Q4_K_M file of 2.02 GB;
- the same prompt length, taken from Ollama's own count;
- the same 128 generated tokens, each appending K/V and attending over the whole context;
- logits back on the host every token.

```powershell
.\scripts\build.ps1 -Preset windows-msvc-nvidia
.\scripts\compare_llm.ps1                       # needs Ollama running with llama3.2 pulled
```

Results (i5-13400F with 16 logical CPUs, RTX 4060; prompt 43 tokens, 128 generated):

| Run | ms/token | tokens/s | weights GB/s |
|---|---|---|---|
| pcuda, `booth` path (Booth's x86 code), 16 threads | 1,766 | 0.57 | 1.1 |
| pcuda, `native` path, 16 threads | 216 | 4.6 | 9.3 |
| pcuda, `simd` path, 16 threads | 156 | 6.4 | 12.9 |
| pcuda, `simd` path, 8-row interleaved Q4 (`q4x8`) | 111 | 9.0 | 18.0 |
| pcuda, `simd` path, `q4x8` + 8-bit activations, AVX-VNNI (`q4x8q8`) | 51–56 | 18–19.5 | 36–39 |
| pcuda, `simd` path, `q4x8q8` + fused launches | 46–49 | 20–22 | 41–44 |
| pcuda, `simd` path, + affine lanes, transposed K, pool on P-cores | 36–37 | 27.1–27.9 | 54–56 |
| pcuda, `simd` path, + each token as one graph | 33.1–34.0 | 29.4–30.2 | 59–61 |
| pcuda, `simd` path, + software prefetch, parallel norm (default) | **31.8–33.2** | **30.1–31.4** | **60–63** |
| CUDA, same kernels (nvcc) | 65 | 15.4 | 30.9 |
| CUDA, same kernels, `q4x8` (coalesced) | 39 | 25.7 | 51.5 |
| CUDA, same kernels, `q4x8q8` (`__dp4a`), transposed K, CUDA graphs | 33 | 30.2 | 60.7 |
| CUDA, same kernels, + parallel norm | 20.4 | 49.0 | 98.3 |
| plain C++ `/O2`, 1 thread | 1,195 | 0.84 | 1.7 |
| Ollama llama.cpp, CPU only (7 runs) | 34–38 | 26.2–29.5 | 53–60 |
| Ollama llama.cpp, GPU | 9.3 | 107 | 216 |

- Compiling Booth's IR with MSVC instead of using Booth's x86 back end makes the same kernels 8× faster
  (`native`). Running 8 CUDA threads per AVX2 register makes them 11× faster (`simd`). All three
  pcuda paths give bit-identical logits, within 4e-7 of the C++ reference.
- Same source on the GPU is still 2.4× faster than `simd`. Real llama.cpp is 4.5× faster again on
  this CPU and 7× on the GPU. Its kernels use techniques this kernel subset can't express: integer
  dot products on quantized activations, and warp-cooperative, coalesced loads.
- `q4x8` (`kernels/llm/matvec_q4x8.cu`) stores 8 rows' bytes side by side, so the 8 SIMD lanes
  read 8 adjacent bytes with one load and neighbouring GPU threads make coalesced loads. It also sums
  each block in 4 independent partial sums, so a core isn't stalled on one chain of dependent
  additions. It is still one-thread-per-row CUDA C that Booth accepts, and all paths stay
  bit-identical. Its results differ from `q4` only by summation order.
- `q4x8q8` quantizes each matrix-vector input to 8 bits per block of 32, as llama.cpp does
  (`quantize_q8x4`). The weights are multiplied with integer dot products through a small
  `pcuda_dp4a` helper:
  - the simd path uses AVX-VNNI `vpdpbusd`;
  - CUDA uses `__dp4a`;
  - Booth and native compute the same bytes the slow way.
  Integer math is exact, so all CPU paths stay bit-identical. Logits move by the activation
  rounding only (same top-5 tokens).
- Fused launches: q/k/v and gate/up are single matrices (one launch, one activation
  quantization), SwiGLU is fused with quantization, and each residual add with the next RMS norm.
  That's about 15 launches per layer instead of 24, with bit-identical results. Launches with fewer
  than 8 threads use the native path.
- Then, with the same kernels and memory traffic, simd went from 46–49 to 36–37 ms per token. It is
  now on par with Ollama on CPU: in two back-to-back runs, 27.1 vs 26.2 and 27.5 vs 29.5 tok/s.
  - The generator keeps lane-affine values scalar (see Code paths): the matvec inner loop is now
    one vector load and two `vpdpbusd`, with no per-load checks.
  - VNNI is picked once per launch (a template parameter). A check inside the loop had made MSVC
    spill every vector register.
  - K is cached transposed (`kv_append` writes it), so the 8 lanes of `attn_scores` read 8
    consecutive positions. `pcuda_aligned8` keeps head and position indices uniform.
  - The pool hands out chunks through an atomic counter and wakes workers with one
    `WakeByAddressAll`. Waiting threads use `UMWAIT` where available.
  - The pool uses only the performance cores' logical processors, minus one for the launching
    thread: 11 threads here. With 16 threads plus the launching and host threads, waiting workers
    took processors from the threads with work, and busy efficiency cores slowed the rest. 12
    threads in all ran 10–15% faster than 18.
- The generator then added software prefetch: a prefetch 2 KB ahead of every contiguous load inside a
  loop, because the hardware prefetcher stops at 4 KB pages. That saves about 1–1.5 ms per token.
  Residual add + RMS norm + 8-bit quantization now runs as two parallel kernels (`add_sumsq`,
  `norm_q8x4`) instead of a one-thread kernel plus a separate quantize, and rope covers q and k in one
  launch. That helps pcuda little, since graphs already made those steps cheap. It helps CUDA a lot:
  its one-thread norm had cost 13 ms per token, so CUDA went from 33 to 20 ms. Against Ollama:
  30.7 vs 28.3 tok/s.
- Two rows per thread (`matvec_q4x8_q8x2`, `--rows-per-thread 2`): each thread works on two
  8-row groups at once, with two weight streams and four dot-product chains. Results are
  bit-identical, but it was no faster: about 0.5 ms slower on the CPU and the same on CUDA. With 12
  threads plus hardware and software prefetch, memory is already kept busy, so it stays off by
  default.
- Each token now runs as one pcuda graph: about 450 launches recorded, then run as one pool job,
  instead of a stream hand-off and pool wake-up per launch. That saves about 4 ms per token
  (37.5 → 33.6 ms in alternating runs, `--graphs on|off`). In two runs against Ollama, pcuda was
  ahead: 29.4 vs 28.6 and 30.2 vs 28.7 tok/s. The CUDA side uses CUDA graphs in the same way,
  as llama.cpp does, but gains little: its time is in the kernels.
- This machine reads memory at up to 84 GB/s (DDR5-6000, 16 threads), a limit of about 41.6 tok/s for
  2.02 GB per token. Ollama reaches 58 GB/s, and more threads don't change that
  (6/10/16 threads: 26.4/29.5/28.6 tok/s).
- Slow paths time an evenly spaced subset of the 128 positions (the "tokens timed" column in the
  bench output), so the average covers the same context range.

## Runtime settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PCUDA_GUARD_PAGES` | `1` | `0` uses `_aligned_malloc` instead of guard-page allocations |
| `PCUDA_KERNEL_STACK_MB` | `8` | Stack reserve for stream and worker threads (1–1024) |
| `PCUDA_CODEGEN` | `auto` | Kernel code path: `auto`, `booth`, `native` or `simd` (see Code paths) |
| `PCUDA_POOL_THREADS` | P-core logical CPUs − 1 | Threads in the chunked-launch pool (1–1024); the launching thread runs chunks too. On non-hybrid CPUs: all logical CPUs − 1 |
| `PCUDA_CHUNKS_PER_WORKER` | `4` | A chunked launch with `workers = N` is split into up to N × this many chunks, so fast cores pick up the slow (efficiency) cores' share |
| `PCUDA_PREFETCH_BYTES` | `2048` | How far ahead the simd path prefetches streaming loads (0 = off; 1.5–8 KB measured the same) |
| `PCUDA_LARGE_PAGES` | `0` | `1` backs allocations of 2 MB or more with large pages, for fewer TLB misses. Needs the "Lock pages in memory" right: run `scriptsgrant_lock_pages.ps1` as administrator once, then sign out and in. Without it, normal pages are used. Overruns still fault (a reserved no-access range follows each block). Measured: about 1 ms per token faster with prefetch off, no gain on top of the default prefetch |
| `PCUDA_POOL_SPIN_US` | `50` | Pool and stream threads, and a waiting caller, wait this long (in `UMWAIT` if the CPU has WAITPKG, else spinning) before sleeping, so back-to-back launches skip the thread wake-up. Longer was slower here |
| `PCUDA_ALLOW_EMULATION` | unset | `1` lets the runtime start under x64 emulation on ARM64 (unsupported) |

## Limits and known issues

- Executables are linked with `/CETCOMPAT:NO`. Fault recovery rewrites RSP/RIP, which hardware shadow
  stacks would reject.
- Processes under an Arbitrary Code Guard policy get `Error::DynamicCodeBlocked`.
- Host funcs (`pcudaLaunchHostFunc`) must not call pcuda APIs. Doing so returns `UnsupportedOperation`.
- `pcudaStreamWaitEvent` (cross-stream waits) is not implemented.
- The `simd` path runs 8 CUDA threads in lock step instead of one after another. Kernels that
  communicate between threads through memory without synchronization (a data race in CUDA too) can
  give different results on it.
- ThreadSanitizer isn't available on Windows. Chunk disjointness is checked by the runtime on every
  chunked launch and by tests.
