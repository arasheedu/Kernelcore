# CUDA CPU Runtime POC (Windows x64): Booth CPU Backend + PantheonSim-Inspired Runtime

**Status:** Revision 2 (Windows-native), **implemented on 2026-11-06**: POC 0.1 and 0.2 build and pass all tests on Windows 11 x64 with MSVC 14.44. §18 records the results and where the implementation departs from this text. Revision 1 (Linux/WSL) is kept as `cuda_cpu_poc_plan.v1-linux.md`. The rationale for every change is in `cuda_cpu_poc_plan_review.md`.  
**Target:** A small, testable proof of concept, not a full CUDA replacement  
**Primary platform:** Windows 11 / Windows 10 22H2+, x64, native (no WSL needed at runtime)  
**Toolchain:** Visual Studio 2022 Build Tools (MSVC v143, `ml64`) **or** LLVM `clang-cl` ≥ 17 with the MSVC STL; CMake ≥ 3.26; Ninja (recommended) or the VS generator  
**Primary language:** C++20 host runtime, x64 MASM for the ABI thunk, CUDA C kernel source  
**Compiler backend:** Booth `kath --cpu` (build-time tool only)  
**Runtime design reference:** PantheonSim / VirtualGPU architecture

---

## 1. Objective

Build a small runtime that accepts a deliberately limited subset of CUDA-style code, compiles CUDA kernel source to x86-64 machine code with Booth, and executes that code **natively on Windows** through a PantheonSim-inspired host runtime.

The POC must demonstrate:

1. A CUDA `.cu` kernel compiles with Booth's CPU backend without an NVIDIA GPU.
2. A Windows host application can discover a CPU device, allocate device-style memory, copy data, launch a registered kernel, synchronize, and retrieve results.
3. A Booth-generated System V object runs correctly inside a Win64 process, through a validated loader and an ABI thunk.
4. A faulting kernel is **contained**: it reports an error and does not crash the process silently.
5. A backend boundary that could later host a native NVIDIA CUDA backend.
6. Optionally, a chunked 1-D experiment over N CPU workers, documented as host-side decomposition only.

**Success means functional proof, robustness and clean architecture. GPU-equivalent performance and full CUDA compatibility are not goals.**

## 2. Upstream facts this plan depends on

Verified against Booth `master` on 2026-11-06. **Phase 0 re-verifies every row against the pinned commit; any mismatch is a stop-and-replan event.**

| ID | Fact | Source | Consequence |
|---|---|---|---|
| F1 | `--cpu` writes **ELF64 relocatable** objects only. There is no COFF/PE output. | `src/cpu/cpu_elf.c` | Windows linkers can't consume it, so the runtime loads it itself (§7). |
| F2 | Sections: NULL, `.text`, `.rela.text`, `.shstrtab`, `.symtab`, `.strtab`. No `.data`, `.rodata`, `.bss` or `.eh_frame`. | `cpu_elf.c` | The loader is small: one code section. |
| F3 | The only relocation type is `R_X86_64_PLT32` (addend −4), used for external calls. | `cpu_elf.c` | The loader supports this one type and rejects everything else. |
| F4 | Possible externals: `sinf`, `cosf`, `exp2f`, `log2f`. | `cpu_emit.c` | Resolve only from an allowlist of System V shims (§7.4). |
| F5 | Calling convention is **System V AMD64**. `rsi`, `rdi` and `xmm6–15` are not preserved. No red zone is used. The stack is aligned to 16 before calls. | `cpu_emit.c` | A MASM thunk is required (§7.3). |
| F6 | Hidden trailing `nthreads` is compared as a **64-bit** value. | `cpu_emit.c` | Pass every integer argument as a sign-extended 64-bit value. Never declare it as `int`. |
| F7 | Only the **first** `__global__` function in a file is emitted. The symbol is the unmangled source name, `STB_GLOBAL STT_FUNC`. | `cpu_elf.c` | One kernel per `.cu` file. Look up by exact name. |
| F8 | `threadIdx.x` runs from 0 to `nthreads`−1. `blockIdx` = 0, `gridDim` = 1, `blockDim` = `nthreads`. | `cpu_emit.c` | Single block per call. The chunking contract follows from this (§11). |
| F9 | Codegen is correctness-first with no register allocator and everything on the stack. `tl.load` masks aren't honoured. | `docs/features.md` | Frames are large. Workers need big stacks. Set `nthreads` exactly. |
| F10 | Booth builds with plain C99 `make`. Release notes mention Linux/macOS/Windows binaries with `SHA256SUMS`. | README, releases | There are several ways to obtain `kath` (§13.2). |

## 3. Non-goals for POC 0.1

- Full CUDA Runtime/Driver API or ABI compatibility; no `libcudart` replacement.
- Running PTX, cubins, fatbins or arbitrary precompiled CUDA binaries.
- Translating `kernel<<<grid, block>>>` launch syntax.
- Multi-dimensional grid/block semantics, `__shared__`, `__syncthreads`, warp intrinsics, cooperative groups, device-side launch.
- A general-purpose ELF loader. The loader accepts **only** Booth's documented subset (F1–F4) and rejects everything else.
- Linking Booth objects with `link.exe`/`lld-link` or converting them with `objcopy`. This was rejected because it doesn't fix the ABI (F5) and lets CRT shadow-space writes corrupt the kernel's stack (see the review).
- 32-bit Windows and ARM64 Windows. On ARM64, x64 emulation isn't a supported execution path for this POC; the runtime returns `UnsupportedOperation`.

## 4. Architecture

```text
   kernels/vector_add.cu  (one __global__ per file, LF line endings)
              |
              |  build time: kath --cpu   (exe | wsl | prebuilt; see 13.2)
              v
   vector_add.o  (ELF64, System V)  --->  pcuda_objcheck (build-time validator)
              |
              |  CMake embeds the bytes and SHA-256 into a generated .cpp
              v
+------------------------------------------------------------+
|                 POC Host Application (.exe)                |
|   typed kernel adapters  ->  pcuda* runtime API            |
+-----------------------------+------------------------------+
                              v
+------------------------------------------------------------+
|                      Runtime Core                          |
|  DeviceManager | AllocationRegistry | Error + sticky state |
|  Stream/Event  | Launch validation  | Diagnostics          |
+-----------------------------+------------------------------+
                              v
+------------------------------------------------------------+
|                       Backend API                          |
+--------------+------------------------------+--------------+
               v                              v
          CpuBackend                    NvidiaBackend (stub)
               |
   +-----------+-------------------------------------------+
   | BoothImageLoader  (ELF subset -> VirtualAlloc, W^X)   |
   | SysV shim table   (sinf/cosf/exp2f/log2f, near code)  |
   | pcuda_sysv_invoke (MASM Win64 -> SysV thunk)          |
   | KernelFaultGuard  (vectored exception handler)        |
   | WorkerPool        (8 MB stacks, optional chunking)    |
   +-------------------------------------------------------+
```

### Design rules

1. Runtime core doesn't include Booth compiler internals or NVIDIA headers.
2. **All System V/ELF knowledge lives in `src/cpu/booth_image.*`, `src/cpu/sysv_invoke.asm` and `src/cpu/sysv_shims.asm`.** Nothing else knows the kernel is not native Win64.
3. Kernel code is **never** called through a C++ function pointer. It is called only through `pcuda_sysv_invoke`.
4. Every operation returns an explicit `pcuda::Error`. There is no silent success and no silent fallback to a different implementation.
5. Untrusted input (object bytes, user pointers, sizes) is bounds-checked before use, with overflow-safe arithmetic.
6. Memory ownership is tracked by the runtime, never inferred from pointer values.
7. No exceptions cross the `pcuda*` API boundary or the thunk. Every API entry point is `noexcept` and catches internally.

## 5. Repository layout

```text
cuda-cpu-poc/
├── CMakeLists.txt
├── CMakePresets.json               # windows-msvc, windows-clangcl, *-asan
├── .gitattributes                  # *.cu *.cmake *.sh eol=lf; *.asm *.ps1 eol=crlf; *.o binary
├── README.md
├── BOOTH_PIN.txt                   # tag, commit, kath.exe SHA-256, verified facts F1–F10
├── cmake/
│   ├── BoothKernel.cmake           # booth_add_kernel(): compile → validate → embed
│   └── EmbedBinary.cmake
├── kernels/
│   ├── vector_add.cu
│   └── prebuilt/                   # vector_add.o + vector_add.o.sha256 (committed fallback)
├── include/pcuda/
│   ├── api.hpp  types.hpp  error.hpp  backend.hpp  kernel_registry.hpp
├── src/
│   ├── runtime.cpp  device_manager.cpp  allocation_registry.cpp
│   ├── stream.cpp   event.cpp           kernel_registry.cpp
│   └── cpu/
│       ├── cpu_backend.cpp
│       ├── booth_image.hpp/.cpp    # ELF subset loader
│       ├── sysv_invoke.asm         # Win64 → SysV thunk + recovery label
│       ├── sysv_shims.asm          # SysV → Win64 libm shims
│       ├── fault_guard.cpp         # vectored exception handler
│       ├── guarded_alloc.cpp       # VirtualAlloc + guard page
│       └── worker_pool.cpp
├── adapters/
│   └── vector_add_adapter.cpp
├── tools/
│   └── objcheck.cpp                # pcuda_objcheck: same parser as runtime, CLI
├── app/
│   └── main.cpp
├── tests/
│   ├── harness.hpp                 # tiny in-repo test harness; no network fetches
│   ├── test_memory.cpp  test_loader.cpp  test_thunk_abi.cpp
│   ├── test_vector_add.cpp  test_fault.cpp  test_stream.cpp
│   ├── test_chunked_vector_add.cpp
│   └── loader_corpus/              # malformed ELF files: truncated, bad reloc, bad machine, ...
└── scripts/
    ├── bootstrap.ps1               # checks toolchain, fetches/validates kath, prints a report
    ├── build_kath_wsl.sh           # fallback: build pinned kath inside WSL
    └── run_poc.ps1
```

## 6. Runtime API for POC 0.1

The `pcuda*` names avoid collisions with CRT macros (`malloc`/`free` under `_CRTDBG_MAP_ALLOC`) and with NVIDIA headers.

```cpp
namespace pcuda {

enum class Error : int {
    Success = 0,
    NotInitialized,
    InvalidValue,
    InvalidDevice,
    OutOfMemory,
    InvalidPointer,          // not a live registered allocation / range out of bounds
    UnsupportedOperation,
    InvalidKernelImage,      // loader/validator rejected the object
    DynamicCodeBlocked,      // process mitigation (ACG) forbids executable memory
    LaunchFailure,           // launch rejected before execution
    KernelFault,             // access violation / divide-by-zero inside kernel (sticky)
    ContextPoisoned,         // any call after a sticky fault, until pcudaDeviceReset
    NotReady
};

enum class CopyKind { HostToDevice, DeviceToHost, DeviceToDevice, HostToHost };

struct DeviceProperties {
    char     name[128];          // e.g. "pcuda CPU backend (Booth x86-64)"
    unsigned worker_count;
    size_t   total_memory;       // GlobalMemoryStatusEx ullTotalPhys
    bool     is_hardware_gpu;    // always false for CPU backend
};

struct FaultInfo {               // filled after KernelFault
    unsigned long exception_code;// EXCEPTION_ACCESS_VIOLATION, ...
    uintptr_t     fault_address; // data address for AV
    uintptr_t     code_offset;   // RIP offset inside kernel .text
    char          kernel[64];
};

} // namespace pcuda

extern "C++" {
pcuda::Error pcudaInit() noexcept;
pcuda::Error pcudaShutdown() noexcept;            // explicit; never rely on static dtors
pcuda::Error pcudaGetDeviceCount(int* count) noexcept;
pcuda::Error pcudaGetDeviceProperties(pcuda::DeviceProperties*, int device) noexcept;
pcuda::Error pcudaSetDevice(int device) noexcept;
pcuda::Error pcudaMalloc(void** ptr, size_t bytes) noexcept;
pcuda::Error pcudaFree(void* ptr) noexcept;
pcuda::Error pcudaMemcpy(void* dst, const void* src, size_t bytes, pcuda::CopyKind) noexcept;
pcuda::Error pcudaDeviceSynchronize() noexcept;
pcuda::Error pcudaDeviceReset() noexcept;         // clears ContextPoisoned, frees everything
pcuda::Error pcudaGetLastFault(pcuda::FaultInfo*) noexcept;
const char*  pcudaGetErrorString(pcuda::Error) noexcept;
const char*  pcudaGetLastErrorDetail() noexcept;  // thread-local diagnostic text
}
```

### Semantics (all specified, none left to the implementation)

- One logical CPU device (`0`). Any other index returns `InvalidDevice`.
- Calling any API before `pcudaInit`, or after `pcudaShutdown`, returns `NotInitialized`. `pcudaInit` and `pcudaShutdown` themselves are idempotent.
- `pcudaMalloc(p, 0)` returns `InvalidValue`. `bytes` is rounded up with overflow checks, and an overflow returns `OutOfMemory`.
- `pcudaFree(nullptr)` returns `Success` (no-op). Unknown, interior and double-freed pointers return `InvalidPointer`.
- `pcudaMemcpy` with `bytes == 0` returns `Success` without touching pointers. Each device operand's `[ptr, ptr+bytes)` range must lie inside one live allocation (interior pointers allowed). `DeviceToDevice` uses `memmove` semantics. `HostToHost` validates only for null.
- After a `KernelFault`, work-creating calls (malloc, memcpy, memset, launch, stream/event create and record) and synchronize calls return `ContextPoisoned`. Release calls (`pcudaFree`, stream/event destroy), queries, `pcudaGetLastFault`, `pcudaDeviceReset` and `pcudaShutdown` still work, so callers can clean up. An async fault is first reported as `KernelFault` by the stream's synchronize, then `ContextPoisoned`.
- `pcudaGetLastErrorDetail()` holds text from the most recent pcuda call on that thread. Every call resets it.
- Host funcs must not call pcuda APIs. Calls from runtime worker threads return `UnsupportedOperation` instead of risking a shutdown deadlock.
- The device is reported as a CPU backend and never as an NVIDIA GPU.

## 7. Executing Booth objects on Windows

### 7.1 Kernel source contract

```cpp
// kernels/vector_add.cu  — exactly one __global__ per file (F7)
extern "C" __global__ void vector_add(const float* a, const float* b,
                                      float* out, int count)
{
    const int i = threadIdx.x;      // blockIdx.x == 0 on CPU backend (F8)
    if (i < count) out[i] = a[i] + b[i];
}
```

Contract enforced by `pcuda_objcheck` and review:
- At most **5** declared integer/pointer parameters (+ hidden `nthreads` = 6 GPRs) and at most **8** float parameters. No struct-by-value parameters.
- No `__shared__`, `__syncthreads`, atomics or warp intrinsics.
- Phase 0 confirms whether Booth accepts `extern "C"`. If it doesn't, drop it; Booth doesn't mangle names (F7).

### 7.2 `BoothImageLoader` (`src/cpu/booth_image.cpp`)

Input: a `std::span<const std::byte>` (embedded bytes). Output: `BoothImage` with `.text` mapped and the kernel entry resolved, or `InvalidKernelImage` with a precise reason.

Validation, in order. All offset+size checks use overflow-safe helpers.
1. `size ≥ sizeof(Elf64_Ehdr)`. `EI_MAG0..3` = `\x7fELF`, `EI_CLASS = ELFCLASS64`, `EI_DATA = ELFDATA2LSB`, `e_type = ET_REL`, `e_machine = EM_X86_64 (62)`, `e_shentsize = 64`.
2. The section header table lies inside the buffer. `e_shstrndx` is valid. Every section's `[sh_offset, sh_offset+sh_size)` lies inside the buffer (except `SHT_NOBITS`).
3. The section set ⊆ {NULL, `.text`, `.rela.text`, `.shstrtab`, `.symtab`, `.strtab`}. Anything else is rejected, which flags a Booth version drift.
4. The symbol table's `sh_entsize = 24`. Every `st_name` is NUL-terminated inside `.strtab`.
5. Every relocation in `.rela.text` has type `R_X86_64_PLT32 (4)`, `r_offset + 4 ≤ text_size`, a valid symbol index, and an undefined symbol whose name is in the shim allowlist (§7.4).
6. Exactly one defined `STB_GLOBAL STT_FUNC` symbol whose name equals the expected kernel name and whose `st_value < text_size`.

Mapping:
1. `VirtualAlloc(MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE)` one block: `[ .text | pad to 16 | shim jump table ]`. Everything is within ±2 GB, so `rel32` always reaches it.
2. Copy `.text`. Build the jump table: for each used external, a 14-byte stub `jmp qword ptr [rip+0]; dq <address of SysV shim>`.
3. Apply relocations: `value = S + A − P` where S is the stub address. Check that it fits in `int32` before writing (it always will, but check anyway).
4. `VirtualProtect(PAGE_EXECUTE_READ)`, then `FlushInstructionCache`. The memory is **never** writable and executable at the same time.
5. If `VirtualProtect` fails with `ERROR_DYNAMIC_CODE_BLOCKED` (Arbitrary Code Guard policy), return `DynamicCodeBlocked` with a diagnostic. Don't retry and don't fall back.
6. Register the `[base, base+size)` range with `KernelFaultGuard`.
7. Unload: unregister, then `VirtualFree(MEM_RELEASE)`. This is only allowed when no launch is in flight (ref-counted).

The same parser builds `tools/objcheck.cpp` (`pcuda_objcheck <obj> --kernel <name>`). It runs at **build time**, so a bad object fails the build and never reaches runtime. ELF objects carry no parameter signature, so the argument-count limit is enforced by `ArgPacker` when arguments are marshalled, not by objcheck.

### 7.3 `pcuda_sysv_invoke` — Win64 → System V thunk (`src/cpu/sysv_invoke.asm`)

```cpp
// C++ declaration
extern "C" int pcuda_sysv_invoke(const void* entry,          // rcx
                                 const uint64_t gpr[6],       // rdx: rdi,rsi,rdx,rcx,r8,r9
                                 const double   xmm[8],       // r8 : xmm0..xmm7 (low lane)
                                 void**         anchor_slot); // r9 : recovery anchor (thread-local)
// returns 0 = kernel returned normally, 1 = recovered from fault
```

Requirements, each with a test in `test_thunk_abi.cpp`:
- Written as a MASM `PROC FRAME` with `.pushreg`, `.savexmm128` and `.allocstack`, so the thunk has valid Win64 unwind data.
- Saves and restores every Win64 non-volatile register: `rbx, rbp, rdi, rsi, r12–r15, xmm6–xmm15`. `rdi`/`rsi`/`xmm6–15` matter because Booth clobbers them (F5).
- Stack: 8 pushes plus `sub rsp, 168` gives `rsp ≡ 0 (mod 16)` before `call`, as System V requires. `movaps` saves are aligned.
- Stores the post-prologue `rsp` in `*anchor_slot` before calling the kernel.
- Loads `rdi, rsi, rdx, rcx, r8, r9` from `gpr[]` and `xmm0–7` from `xmm[]`, then `call` the entry.
- Has a second label, `pcuda_sysv_recover`, **inside the same PROC after the prologue**. It sets `eax = 1` and jumps to the shared epilogue. The fault guard resumes execution there (§7.5).
- Clears the direction flag (`cld`) before the call. MXCSR and the x87 control word are saved and restored, so a kernel can't leak FP-mode changes.

**Argument marshalling** (`adapters/*`): every integer is **sign-extended to `int64_t`** before being stored in `gpr[]`. This fixes C1/F6 for `nthreads` and protects against Booth loading full 64-bit registers for `int` parameters. Pointers are stored as `uint64_t`. Floats are stored in the low 32 bits of the `xmm` lane as raw bits with `std::bit_cast`, not converted to double.

### 7.4 System V → Win64 libm shims (`src/cpu/sysv_shims.asm`)

For each of `sinf, cosf, exp2f, log2f`:

```asm
pcuda_sysv_sinf PROC FRAME
    sub  rsp, 40          ; 32-byte shadow space + realign (SysV caller left rsp ≡ 8 mod 16 at entry)
    .allocstack 40
    .endprolog
    call sinf             ; Win64 CRT; arg/return already in xmm0 in both ABIs
    add  rsp, 40
    ret
pcuda_sysv_sinf ENDP
```

- Win64 callees preserve a superset of what System V callers expect, so no extra saves are needed.
- Any external not on the allowlist fails `pcuda_objcheck` and the loader.
- Test: a kernel that calls each function is compared against the CRT within 1 ULP.

### 7.5 Fault containment (`src/cpu/fault_guard.cpp`)

- Install one vectored exception handler (`AddVectoredExceptionHandler(1, …)`) in `pcudaInit` and remove it in `pcudaShutdown`.
- Per-thread state (`thread_local`): `in_kernel` flag, `anchor` (from the thunk), current kernel name and image range.
- The handler acts **only** when `in_kernel` is set **and** RIP is inside a registered image or shim range. Otherwise it returns `EXCEPTION_CONTINUE_SEARCH`.
  - Recoverable: `EXCEPTION_ACCESS_VIOLATION`, `EXCEPTION_INT_DIVIDE_BY_ZERO`, `EXCEPTION_INT_OVERFLOW`, `EXCEPTION_ILLEGAL_INSTRUCTION`, `EXCEPTION_PRIV_INSTRUCTION`. Record `FaultInfo`, then set `ctx->Rsp = anchor` and `ctx->Rip = &pcuda_sysv_recover`. Return `EXCEPTION_CONTINUE_EXECUTION`. The thunk restores the non-volatile registers it saved, so the caller's state is intact.
  - `EXCEPTION_STACK_OVERFLOW`: recovered like the others. After the thunk returns, `invokeSysV` calls `_resetstkoflw()` to re-arm the guard page. Only if that fails does the process `__fastfail`. Tests overflow the same thread three times in a row.
- The handler makes no heap allocations, takes no locks and does no CRT I/O.
- After recovery the runtime sets the sticky `ContextPoisoned` state (CUDA-like). Buffers written by the faulted launch are undefined.
- Guard pages make faults deterministic. See §9.

### 7.6 Threads and stacks

- All kernel calls happen on runtime-owned threads created with `_beginthreadex` and an explicit **8 MB** stack (F9), or on the caller's thread for synchronous POC 0.1 launches, where `SetThreadStackGuarantee` reserves room to report a stack overflow.
- Stack size is configurable (`PCUDA_KERNEL_STACK_MB`). The default is 8.
- Worker threads are joined in `pcudaShutdown()`, never in static destructors (avoids the loader-lock deadlock).

## 8. Backend interface

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char*  name() const noexcept = 0;
    virtual pcuda::Error allocate(void** ptr, size_t bytes) noexcept = 0;
    virtual pcuda::Error release(void* ptr) noexcept = 0;
    virtual pcuda::Error copy(void* dst, const void* src, size_t bytes, pcuda::CopyKind) noexcept = 0;
    virtual pcuda::Error launch(const KernelEntry&, const LaunchArgs&) noexcept = 0;
    virtual pcuda::Error synchronize() noexcept = 0;
};

struct LaunchArgs {                 // already marshalled by the typed adapter
    uint64_t gpr[6];  unsigned n_gpr;
    double   xmm[8];  unsigned n_xmm;
    int64_t  nthreads;              // > 0, validated
};

struct KernelEntry {
    const char*                     name;
    std::span<const std::byte>      image;        // embedded ELF bytes
    std::array<uint8_t, 32>         sha256;       // checked at load
    unsigned                        max_int_params;
    // typed adapter: validates user args, fills LaunchArgs, returns Error
    pcuda::Error (*marshal)(const void* typed_args, LaunchArgs* out) noexcept;
};
```

`CpuBackend` is implemented now. `NvidiaBackend` is future work. `-DPCUDA_ENABLE_NVIDIA=ON` (default OFF) currently builds only the side-by-side comparison tool (§19), and no CUDA libraries are linked in the CPU build.

## 9. Allocation registry and guarded allocator

```cpp
struct AllocationInfo {
    std::byte* user_base;   size_t user_bytes;
    std::byte* region_base; size_t region_bytes;   // for VirtualFree
    int device_id;  uint64_t generation;
};
```

- **Allocator** (`guarded_alloc.cpp`): `VirtualAlloc` with `region = round_up(bytes, page) + page`, and the trailing page set to `PAGE_NOACCESS`. Guard mode is **on by default in every configuration**, robustness first. The user block is placed **flush against the guard page** with 64-byte alignment, so for sizes that are multiples of 64 bytes a one-past-the-end write faults immediately and becomes a `KernelFault` instead of silent corruption. Other sizes have up to 63 bytes of undetected slack. `PCUDA_GUARD_PAGES=0` switches to `_aligned_malloc(bytes, 64)` / `_aligned_free`. `std::aligned_alloc` is not used because MSVC doesn't provide it.
- **Registry**: an ordered map keyed by `user_base`, protected by `std::shared_mutex`. Interior-pointer lookup uses `upper_bound` then a range check. Every range check uses overflow-safe `add`.
- Double free and unknown pointers return `InvalidPointer`, and the registry stays unchanged.
- `pcudaShutdown`/`pcudaDeviceReset` free every remaining allocation and report the count of leaked allocations in the diagnostics.

## 10. Streams and events

**POC 0.1:** one default stream, synchronous launch. `pcudaDeviceSynchronize` is a fence that returns the first stored error.

**POC 0.2:**
- `Stream` holds a FIFO and one worker thread (8 MB stack). Operations run in submission order.
- `pcudaStreamSynchronize` waits for the stream to drain and returns the first error from that stream since the last sync.
- `Event` records a completion marker. `pcudaEventSynchronize` waits for it.
- Task errors and C++ exceptions are captured on the worker (`try/catch(...)` → `Error`) and stored. They never propagate across threads.
- No `pcudaMemcpyAsync` until host-buffer lifetime rules are implemented and tested.
- Cross-stream waits are deferred.

## 11. N-core chunking experiment (POC 0.2)

This relies on F8: each call runs one block with `threadIdx.x ∈ [0, nthreads)`, `blockIdx.x = 0`.

1. Split `[0, n)` into `W` disjoint chunks. Each chunk is ≥ 4096 elements and its boundaries are 64-byte aligned, which avoids false sharing on output.
2. For each chunk, the adapter builds a separate `LaunchArgs` with **pointers offset to the chunk start, `count = chunk_len` and `nthreads = chunk_len`**. The original plan left `count` unchanged (review C4).
3. Submit to a persistent `WorkerPool` (W threads, 8 MB stacks). Wait for all chunks. The first error wins. A fault in any chunk poisons the context.
4. Opt-in only, per kernel, via `KernelEntry::chunkable = true`. Never used for reductions, atomics, barriers, or kernels that depend on `blockIdx` or `count`-relative indexing.

## 12. Implementation phases

### Phase 0 — Upstream spike and go/no-go (1–2 days)

1. Pin a Booth tag and commit (start with `v0.6.1`) in `BOOTH_PIN.txt`.
2. Obtain `kath` using the first method that works (§13.2) and record which one in `BOOTH_PIN.txt`. Verify the SHA-256 of any downloaded binary against the release's `SHA256SUMS`.
3. `kath --version`. Compile `kernels/vector_add.cu` with `--cpu` (with and without `extern "C"`).
4. Inspect the object with `llvm-readelf -h -S -s -r` (from the LLVM install) or the first `pcuda_objcheck` prototype. **Re-verify F1–F8.** Record the results.
5. Write a ~150-line spike: a minimal loader, the MASM thunk and a vector-add call with n = 1, 7, 257. Compare against a scalar reference.

**Go/no-go gate.** Continue only if all of these hold:
- the object matches F1–F4,
- the thunk call returns correct results for all three sizes,
- the ASan build of the spike is clean.

If F1–F4 drift (new sections or relocation types), extend the loader allowlist only if the new item is fully specified. Otherwise pin an older commit. If the thunk fails, stop and replan. Don't continue on assumptions.

### Phase 1 — Runtime skeleton (2–3 days)
- CMake project, presets, `.gitattributes`, warnings-as-errors (`/W4 /WX /permissive- /EHsc /utf-8 /Zc:__cplusplus`).
- `Error`, `DeviceManager`, `IBackend`, `CpuBackend` memory paths, guarded allocator, registry.
- Tests: alloc/free/copy, zero sizes, null, double free, interior pointer, out-of-range copy, overlapping D2D, overflow sizes, use before init.

**Exit:** memory tests pass on both MSVC and clang-cl, with ASan on.

### Phase 2 — Loader, thunk, shims and build integration (3–4 days)
- `booth_image.cpp`, `objcheck.cpp`, `sysv_invoke.asm`, `sysv_shims.asm`, `fault_guard.cpp`.
- `booth_add_kernel()` CMake function: run `kath` (by the chosen method) → `pcuda_objcheck` → SHA-256 → embed as `alignas(16) const unsigned char[]` in a generated `.cpp`. Any failure stops the build with a clear message.
- `test_loader.cpp`: every file in `loader_corpus/` must return `InvalidKernelImage` with the expected reason and must never crash. Add a fuzz target (`/fsanitize=fuzzer` with clang-cl) when available.
- `test_thunk_abi.cpp`: hand-written System V test functions in asm that deliberately clobber `rbx, rbp, rdi, rsi, r12–15, xmm6–15`, MXCSR and DF. Assert the caller's state is intact afterwards. Test that 64-bit `nthreads` with garbage upper bits is impossible through the marshaller.

**Exit:** a clean build from an empty build directory with **no** CUDA Toolkit, NVIDIA driver, WSL or network (prebuilt mode) compiles, validates, embeds and runs the kernel.

### Phase 3 — End-to-end launch and fault containment (2–3 days)
- `pcudaLaunchVectorAdd` through the registry, with pointer-ownership checks before launch and a `count ≤ allocation size / sizeof(float)` check.
- Sizes 1, 7, 256, 257, 4096, 1 << 24. Compare bit-for-bit against a scalar reference (float add is exact here).
- `test_fault.cpp`: a deliberately out-of-bounds kernel (`out[i + count]`) with guard pages must return `KernelFault` with `FaultInfo` filled. Afterwards calls return `ContextPoisoned`, `pcudaDeviceReset` recovers, and the process stays alive. Run 1,000 iterations to catch state leaks.
- Stack overflow test (separate process via CTest `WILL_FAIL`/exit-code check): verifies the fast-fail diagnostic.

**Exit:** the app uses only `pcuda*` APIs, results are correct, and faults are contained.

### Phase 4 — Streams/events (2–4 days)
As in §10. Tests cover ordering with host marker tasks, two independent streams, error propagation to the synchronizing caller, and shutdown with pending work.

### Phase 5 — N-core chunking (3–5 days)
As in §11. Correctness on 1, 2, 4 and `std::thread::hardware_concurrency()` workers, compared against the serial reference.
- **Race checking:** ThreadSanitizer isn't available on Windows. Instead use (a) a deterministic disjointness assertion on chunk ranges, (b) guard-page fences between chunk outputs in a test-only mode, and (c) an **optional** WSL/Linux CI job that builds the portable core (registry, worker pool, chunk planner) with TSan.
- Benchmarks: `std::chrono::steady_clock` (QPC-backed), 5 warm-up runs then a median of 21, power plan recorded, optional `SetThreadAffinityMask`. Report scheduler overhead separately from kernel time.

### Phase 6 — Optional NVIDIA parity (later)
Same as revision 1. It must not affect the CPU-only build.

## 13. Build and run on Windows

### 13.1 Prerequisites (checked by `scripts/bootstrap.ps1`)

| Item | Check | Required |
|---|---|---|
| Windows x64 | `$env:PROCESSOR_ARCHITECTURE -eq 'AMD64'` | Yes |
| VS 2022 Build Tools (C++ workload) | `vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` | Yes |
| `ml64.exe` | in the VS Developer environment | Yes |
| CMake ≥ 3.26, Ninja | `cmake --version`, `ninja --version` | Yes |
| LLVM (clang-cl, llvm-readelf) | `clang-cl --version` | Optional (second compiler, Phase 0 inspection, fuzzing) |
| `kath` | §13.2 | Only if not using prebuilt mode |
| WSL | `wsl --status` | Optional fallback only |

The script prints a pass/fail table and exits non-zero on any required failure. It never installs anything without being asked.

### 13.2 Obtaining the kernel object (`-DPCUDA_KATH_MODE=auto|exe|wsl|prebuilt`)

`auto` (default) tries the methods in order and logs which one it used:
1. **exe:** `KATH_EXECUTABLE` or `kath.exe` on PATH. Its SHA-256 must match `BOOTH_PIN.txt`. A mismatch is a warning in `auto` and an error in `exe`.
2. **wsl:** `wsl.exe -e sh -c "<kath> --cpu …"` with paths translated by `wslpath`. `scripts/install_kath_wsl.sh` installs the pinned Linux release (SHA-256 checked), or builds the pinned commit with `--from-source`. Booth's output is byte-identical regardless of host (verified).
3. **prebuilt:** `kernels/prebuilt/<name>.o`, used only if its SHA-256 matches the committed `.sha256` and the `.cu` file's hash matches the one recorded with it. This means a stale prebuilt object can't be used silently after the source changes.

If every method fails, configuration fails and names each method and why it failed. After any successful `kath` run, `cmake --build --target refresh_prebuilt` updates the prebuilt object and its hashes.

### 13.3 Commands (PowerShell, from a VS x64 Developer shell)

```powershell
# 0. Environment report
pwsh scripts/bootstrap.ps1

# 1. Configure + build (kernel compile/validate/embed happens inside the build)
cmake --preset windows-msvc            # or windows-clangcl / windows-msvc-asan
cmake --build --preset windows-msvc

# 2. Run
.\build\windows-msvc\pcuda_poc.exe

# 3. Tests
ctest --preset windows-msvc --output-on-failure
```

Notes:
- Paths with spaces are supported. Every custom command uses `VERBATIM` and list arguments.
- `.cu` files are checked out with LF (`.gitattributes`). The CMake function also normalises CRLF to LF before passing the file to `kath`.
- Control Flow Guard (`/guard:cf`) is safe to enable because kernel code is reached only from the MASM thunk's `call`, which CFG doesn't instrument (rule 3).

### 13.4 CI (GitHub Actions)

| Job | Runner | Mode | Purpose |
|---|---|---|---|
| win-msvc | `windows-2022` | prebuilt | Main gate: build, all tests |
| win-msvc-asan | `windows-2022` | prebuilt | ASan |
| win-clangcl | `windows-2022` | prebuilt | Second compiler, fuzz smoke |
| win-kath | `windows-2022` | exe | Fresh `kath` compile, prebuilt hash comparison |
| linux-tsan (optional) | `ubuntu-24.04` | n/a | TSan on the portable core |

## 14. Test matrix and acceptance criteria

| Area | Test | Pass condition |
|---|---|---|
| Toolchain | `bootstrap.ps1` | All required items pass, or a clear failure is reported |
| Compiler | `kath --cpu vector_add.cu` | ELF object emitted, no GPU |
| Validation | `pcuda_objcheck` | Accepts the good object and rejects every corpus file with the right reason |
| Loader | Map, relocate, set W^X | Never writable and executable at once; ACG case → `DynamicCodeBlocked` |
| ABI | Thunk clobber test | All Win64 non-volatile registers, MXCSR and DF preserved |
| ABI | 64-bit marshalling | `nthreads`/ints sign-extended; garbage upper bits impossible |
| Shims | `sinf/cosf/exp2f/log2f` kernel | ≤ 1 ULP from the CRT |
| Device | Enumerate | One CPU device, `is_hardware_gpu = false` |
| Memory | Alloc/free/copy/edge cases | Correct bytes; invalid operations return the specified errors |
| Kernel | Vector add, sizes 1…1<<24 | Bit-exact against the reference |
| Fault | OOB kernel × 1000 | `KernelFault` then `ContextPoisoned`; reset recovers; process alive |
| Stack overflow | Separate process | Fast-fail with diagnostic, non-zero exit |
| Stream/Event | Ordering, errors, shutdown | Order preserved; first error returned; clean shutdown |
| Chunking | 1/2/4/N workers | Equal to the serial reference; disjointness assert holds |
| Isolation | No CUDA, no WSL, no network | Builds and passes in prebuilt mode |
| Diagnostics | Unsupported operation | Explicit error, never silent success |

### Definition of done

**POC 0.1** is done when:
- A clean Windows checkout builds with MSVC **and** clang-cl in prebuilt mode, and in `exe` mode where `kath.exe` is available.
- Booth is pinned by tag, commit and binary SHA-256, with F1–F10 re-verified.
- No NVIDIA hardware, CUDA Toolkit, driver or WSL is needed.
- The host app uses only `pcuda*` APIs.
- The vector-add, loader, ABI, shim and fault tests pass, including under ASan.
- There's a clean seam for an NVIDIA backend.

**POC 0.2** is done when streams/events and chunking pass, along with the Windows race-check substitutes (and the optional Linux TSan job, if enabled).

## 15. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Booth changes object layout, relocations or ABI | Medium (0.x) | Pin commit and hash; the strict allowlist in loader and objcheck fails the build, not the runtime; Phase 0 re-verifies |
| `kath` unavailable on a dev machine or CI | Medium | Four methods (§13.2); committed prebuilt object with source-hash binding |
| Thunk ABI bug (register or stack corruption) | Medium | Dedicated clobber tests; MASM `PROC FRAME` unwind data; spike before Phase 1 |
| Kernel writes out of bounds | High during development | Guard-page allocator, launch-size validation, VEH containment, sticky poison |
| Stack overflow from stack-heavy codegen | Medium | 8 MB worker stacks, `SetThreadStackGuarantee`, fast-fail diagnostic |
| Dynamic code blocked by policy (ACG, enterprise lockdown) | Low | Detect `ERROR_DYNAMIC_CODE_BLOCKED` → `DynamicCodeBlocked`; documented |
| Antivirus flags executable `VirtualAlloc` memory | Low | W^X discipline; signed builds if distributed; documented |
| Booth's 64-bit param handling differs from assumption | Medium | Sign-extend everything to 64 bits in the marshaller |
| CRT shadow-space corruption from libm calls | Certain without shims | Allowlisted System V shims only |
| No TSan on Windows | Certain | Deterministic disjointness checks, guard fences, optional Linux TSan job |
| Static-destructor/loader-lock deadlock | Medium | Explicit `pcudaShutdown`; no thread joins in static destructors |
| Confusing CPU execution with GPU emulation | Low | Honest device name; `is_hardware_gpu = false`; docs |
| Accidental CUDA dependency | Low | CI job without CUDA; NVIDIA backend off by default |

## 16. Upstream references

- Booth repository: https://github.com/Zaneham/Booth
- Booth releases: https://github.com/Zaneham/Booth/releases
- Booth usage guide: https://github.com/Zaneham/Booth/blob/master/docs/usage.md
- Booth feature status: https://github.com/Zaneham/Booth/blob/master/docs/features.md
- Booth CPU ELF writer: https://github.com/Zaneham/Booth/blob/master/src/cpu/cpu_elf.c
- Booth CPU code emitter: https://github.com/Zaneham/Booth/blob/master/src/cpu/cpu_emit.c
- Booth CPU vector-add example: https://github.com/Zaneham/Booth/blob/master/examples/cpu_launch_vadd.c
- PantheonSim repository: https://github.com/pantheongpu/pantheonsim
- PantheonSim architecture: https://github.com/pantheongpu/pantheonsim/blob/main/ARCHITECTURE.md
- Microsoft x64 calling convention: https://learn.microsoft.com/cpp/build/x64-calling-convention
- x64 prolog/epilog and unwind data: https://learn.microsoft.com/cpp/build/prolog-and-epilog
- System V AMD64 psABI: https://gitlab.com/x86-psABIs/x86-64-ABI

## 17. First implementation decision

Start with the **Phase 0 spike**: one `vector_add.cu`, a minimal ELF loader, the MASM thunk and a scalar reference, all in one throwaway executable. It tests the riskiest assumption (System V object code running safely inside a Win64 process) in about a day. Build the runtime, streams and chunking only after the go/no-go gate passes.

## 18. Implementation status (2026-11-06)

Environment: Windows 11 x64 (16 logical processors), Visual Studio 2022 17.14 / MSVC 14.44, CMake 3.31, Ninja (bundled with VS), Booth 0.6.1.

### Phase results

| Phase | Result |
|---|---|
| 0: spike and go/no-go | **Go.** F1–F9 confirmed against real `kath` output. Two new facts found: F10 (identical objects from Windows, Linux and WSL `kath`) and F11 (sin/cos argument lowering). Details are in `BOOTH_PIN.txt`. The spike went straight into the real loader and thunk. |
| 1: runtime skeleton | Done: `test_api` (7 cases), `test_memory` (11 cases) |
| 2: loader, thunk, shims, build | Done: `test_loader` (8 cases, including 60,000 random mutations), `test_thunk_abi` (6 cases, register-clobber harness in MASM) |
| 3: end-to-end and faults | Done: `test_vector_add` (7 cases, sizes 1 to 2^24, bit-exact), `test_fault` (6 cases, 1,000 contained out-of-bounds faults, divide-by-zero, 3 consecutive stack overflows) |
| 4: streams and events | Done: `test_stream` (8 cases) |
| 5: chunking | Done: `test_chunked_vector_add` (4 cases, 1/2/4/16 workers, bit-exact against serial) |
| 6: NVIDIA parity | Comparison harness done (§19); LLM decode benchmark done (§20). Faster code paths and Ollama comparison: §21. `PCUDA_ENABLE_NVIDIA=ON` builds `pcuda_vs_cuda`; an `NvidiaBackend` behind `IBackend` is still not implemented. |

Verified configurations: MSVC RelWithDebInfo; MSVC Debug + AddressSanitizer; `PCUDA_KATH_MODE` = exe, wsl and prebuilt (prebuilt with `kath` unavailable); a source path containing spaces and parentheses. **Not verified:** clang-cl (not installed on the dev machine) and the GitHub Actions workflow (written but never run).

### Benchmark (`pcuda_bench`, median of 21, AC power)

| Elements | Serial | 4 workers | 16 workers |
|---|---|---|---|
| 1,024 | 3.3 µs | 0.13× | 0.11× |
| 262,144 | 0.65 ms | 2.85× | 2.68× |
| 4,194,304 | 10.0 ms | 3.59× | 6.35× |
| 16,777,216 | 41.3 ms | 3.84× | 5.57× |

Launch overhead: about 1.2 µs synchronous, about 41 µs chunked (pool wake-up). Chunking is worth it only from about 256K elements.

### Departures from the text above

1. **Stack overflow is recoverable** (§7.5 updated). It isn't a fail-fast.
2. **Guard pages default on** in every configuration (§9 updated).
3. **Argument-count limit** is enforced in `ArgPacker`, not `pcuda_objcheck` (§7.2 updated).
4. **Poisoned-context rules** let release calls through (§6 updated).
5. **F11, Booth sin/cos lowering:** `sinf(x)` compiles to `sinf((x * 1/(2π)) * 2π)` in float, so results differ from a direct CRT `sinf(x)` by a few ULP. Tests compare bit-exactly against that exact lowering and loosely against double precision.
6. **Name clash:** `FaultInfo::exception_code` was renamed to `code` because `exception_code` is a macro in `<excpt.h>`.
7. **Layout:** `scripts/run_poc.ps1` became `scripts/build.ps1`, which loads the VS environment itself. `build_kath_wsl.sh` became `install_kath_wsl.sh`. Malformed-ELF cases are generated in `test_loader.cpp` instead of committed under `loader_corpus/`. Fuzzing is a seeded mutation test instead of libFuzzer, since there is no clang-cl.
8. **`/CETCOMPAT:NO`** is set explicitly on all executables, because fault recovery rewrites RSP/RIP.
9. **Kernel threads:** every launch, including synchronous ones, runs on a runtime-owned thread with an 8 MB stack (the default stream's worker). None run on the caller's thread.

## 19. Side-by-side comparison with CUDA (Phase 6, part 1; 2026-11-06)

**Goal:** run the same `.cu` source on pcuda and on a real NVIDIA GPU, check that the results agree and measure the speed difference. It's a comparison harness, not an `NvidiaBackend`: the GPU path calls the CUDA runtime directly.

**Design**
- Kernels use the portable index `blockIdx.x * blockDim.x + threadIdx.x`. Under Booth this reduces to `threadIdx.x` (F12), and on the GPU it is the global index. `vector_add`, `scale_add` and `trig_mix` were converted and their prebuilt objects refreshed. The fault-test kernels are unchanged.
- `gpu/cuda_kernels.cu` `#include`s the same kernel files and launches them with 256-thread blocks. It exposes a plain C++ interface, so `app/compare.cpp` is compiled by `cl` with the project's warnings-as-errors settings.
- `cmake/NvidiaCompare.cmake`: opt-in via `PCUDA_ENABLE_NVIDIA=ON` (preset `windows-msvc-nvidia`). It reads the `nvcc` version before `enable_language(CUDA)`. For CUDA < 12.4 with MSVC ≥ 19.40 it adds `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` and warns. Device code uses default `nvcc` settings: precise `sinf`, FMA contraction on, `CMAKE_CUDA_ARCHITECTURES=native`. It isn't combinable with ASan.
- `scale_add` and `trig_mix` gained chunk hooks, so all three kernels run on all cores. `test_chunked_vector_add` checks them bit-exactly against serial launches without a GPU.
- `pcuda_vs_cuda --check` (a ctest, which skips with exit code 77 when no GPU is present) uses the sizes 1, 7, 255, 256, 257, 1000, 4097, 65537 and 2^20+3:
  - `vector_add`: GPU and pcuda bit-identical.
  - `scale_add`: at most 2 ULP apart.
  - `trig_mix`: both within 4e-6 relative error of double precision.
  - pcuda: chunked results equal serial results.
- Timing: up to 21 samples after warm-up, with about 1 s budget per cell. Before each GPU measurement the kernel runs for 300 ms. Without that the GPU stays at idle clocks after the multi-second CPU measurements, and `trig_mix` at 4M showed 4.5 ms instead of 0.13 ms.

**Results** (i5-13400F with 16 logical processors, RTX 4060 with 24 SMs, CUDA 12.3, driver 13.3)

| Kernel, 16M elements | C++ /O2 | pcuda 1T | pcuda 16T | CUDA kernel | pcuda + copies | CUDA + copies |
|---|---|---|---|---|---|---|
| `vector_add` | 6.2 ms | 47.9 ms | 8.7 ms | 0.80 ms | 18.1 ms | 16.3 ms |
| `scale_add` | 5.2 ms | 46.8 ms | 8.4 ms | 0.89 ms | 16.4 ms | 10.9 ms |
| `trig_mix` | 170 ms | 442 ms | 62 ms | 0.54 ms | 67 ms | 10.9 ms |

Findings:
1. **Kernel only**, the GPU wins by 10–54× for memory-bound kernels. The peak is at 1M elements, where the data fits in the GPU's L2 cache. From 4M up, DRAM bandwidth caps the gain at about 11×. For the compute-bound `trig_mix` the gain is 115–170×.
2. **End to end**, memory-bound kernels are about even (1.0–1.5×), because PCIe copies dominate. Below about 256K elements pcuda is faster: CUDA launch-and-wait costs 9–16 µs and the copies 50–80 µs. `trig_mix` stays about 6× faster on the GPU.
3. **Launch overhead:** pcuda about 1 µs synchronous and about 30 µs chunked; CUDA 9–16 µs (WDDM).
4. **Accuracy:**
   - `vector_add` is bit-identical.
   - `scale_add` differs by 1 ULP in about 5% of elements. `nvcc` contracts `x*s+b` into an FMA (one rounding) and Booth rounds twice, so the GPU is slightly closer to the exact result.
   - `trig_mix` differs by up to 16 ULP; relative error is 8.2e-7 for pcuda and 6.9e-7 for the GPU.
5. **Booth codegen** costs about 10× against the C++ `/O2` loop on one thread for simple kernels. It keeps every value on the stack and doesn't vectorize. For `trig_mix` the gap is 2.6×, because libm calls dominate. Using all 16 cores brings pcuda close to single-threaded C++ for memory-bound kernels.

**Not done:** an `NvidiaBackend` behind `IBackend` (pcuda API calls dispatched to CUDA). Comparisons of shared memory, atomics or multi-block kernels are also not done, since Booth's CPU backend doesn't support them.

## 20. LLM decode benchmark (Phase 6, part 2; 2026-11-06)

**Question:** how would LLM inference (token generation) run on pcuda versus CUDA?

**Why not reuse llama.cpp's CUDA backend:** of the 71 `.cu` files in `ggml-cuda` (in the local llama.cpp fork):
- 32–34 use `__shared__` and `__syncthreads`;
- 12 use warp shuffles;
- 48 use `half`;
- 48 are templated;
- 6 call cuBLAS.

Booth's CPU backend supports none of these. `__shared__` with `__syncthreads` even compiles but gives wrong results, because the threads run one after another. So step 1 builds the decode step from kernels written in the subset both compilers accept.

**Design**
- `kernels/llm/`: seven kernels.
  - `matvec_q8` (int8 blocks of 32 with a float scale each; split arrays, since Booth has no `half`), `rms_norm`, `rope`, `attn_scores`, `softmax`, `attn_values`, `silu_mul`. The residual add is `vector_add`.
  - One thread per output, no early `return`, `exp2f` instead of `expf`.
  - Sizes come in through a device `int` array, because of Booth's limit of 5 integer parameters.
  - Chunked launches pass the first thread index in a `base` parameter where an index doesn't map to a pointer offset (attention).
- `llm/`: pcuda adapters (`llm_kernels.cpp`), a plain C++ reference (`llm_reference.cpp`), and a synthetic model and decode step (`llm_model.cpp`).
  - The model has Llama 3.2 1B/3B shapes and deterministic random Q8 weights, and is built once into both pcuda and CUDA memory.
  - The C++ reference reads pcuda's weights directly.
  - Simplifications: the KV cache is pre-filled and not appended to, and there is no embedding lookup.
  - The adapters check, as with all pcuda kernels, every buffer against live allocations and that the device `dims` array matches the validated sizes.
- `test_llm_kernels` (CPU-only, runs in CI) has 5 cases:
  - `matvec_q8`, both attention kernels and `silu_mul` are bit-identical to C++, and chunked results equal serial ones.
  - `rms_norm` and `softmax` are within 1e-6, and `rope` within 1e-3 (F11 lowering).
  - Argument validation.
  - A whole tiny-model token agrees with C++ (pcuda and pcuda-chunked are bit-identical).
- `pcuda_llm_bench` (NVIDIA build): a per-operation breakdown per token for C++ 1T, pcuda 1T, pcuda NT and CUDA, plus tokens/s, weight bandwidth, and logits agreement with the C++ reference. `--check` (a tiny model) runs as a ctest.
- `scripts/ollama_reference.ps1`: real llama.cpp decode speed through Ollama's API, with the GPU on and off.

**Results** (Llama 3.2 3B shapes: 3.21 B parameters, 3.61 GB of Q8 per token, 512 cached positions)

| ms per token | C++ /O2 1T | pcuda 1T | pcuda 16T | CUDA (same kernels) |
|---|---|---|---|---|
| matvec attention (q, k, v, o) | 269 | 5,004 | 489 | 34.5 |
| matvec FFN (gate, up, down) | 802 | 15,181 | 1,370 | 57.3 |
| matvec lm_head | 137 | 3,246 | 220 | 6.5 |
| attention (scores, softmax, values) | 135 | 627 | 71 | 2.7 |
| rms_norm, rope, silu_mul, add | 2 | 16 | 11 | 10 |
| **total** | **1,344** | **24,074** | **2,161** | **111** |
| tokens/s | 0.74 | 0.04 | 0.46 | 9.0 |

The real-world reference is Ollama 0.40.6 with `llama3.2` (3B, Q4_K_M, 2.02 GB): **107 tok/s on the GPU (216 GB/s) and 27.5 tok/s CPU-only (55 GB/s)**.

**Findings**
1. With the same source, CUDA is about 20× faster than pcuda on 16 threads per token.
2. pcuda on 16 threads is slower than plain C++ on one thread. Booth's code is about 18× slower per thread: it keeps every value on the stack and doesn't vectorize. Matrix-vector products are 95% of the token time.
3. Against real llama.cpp, pcuda reaches 1.7 GB/s of weight traffic versus 55 GB/s for ggml-cpu (33×). Scaled to the 2.02 GB Q4_K_M model, pcuda would manage under 1 tok/s versus 27.5.
4. The same-source CUDA kernels reach 32.6 GB/s, 15% of the RTX 4060's bandwidth, against llama.cpp's 216 GB/s:
   - One thread per row makes neighbouring threads read addresses `k` bytes apart, so loads aren't coalesced.
   - `rms_norm` runs on a single GPU thread, taking 9.5 ms per token.
   The Booth subset rules out the fix: warp-per-row reductions need shuffles or shared memory.
5. Logits agree with the C++ reference to 2.4e-6 (pcuda) and 2.6e-6 (CUDA, FMA contraction), with the same top-5 tokens.

**Step 2 (not started):** a `ggml-pcuda` backend in the llama.cpp fork that runs these kernels for the supported ops and leaves the rest to ggml-cpu. Given finding 3, it would make llama.cpp slower than its own CPU backend. Its value is to show integration, not speed.

## 21. Faster CPU code paths and a like-for-like comparison with Ollama (2026-11-06)

**Goal:** make pcuda's CPU execution fast without giving up Booth as the CUDA compiler, and compare token generation like for like with `ollama run llama3.2`.

**Diagnosis.** Booth's front end and optimizer produce clean SSA (`kath --ir`). Only the x86 back end is weak: it has no register allocator and sends every value through the stack. That makes it about 18× slower per thread than the same loop in C++. Booth's Metal back end prints the same optimized IR as structured C (F13).

**Design**
- **Three code paths per kernel**, all built from the one `.cu` file (`booth_add_kernel`):
  - `booth`: the `--cpu` object, as before.
  - `native`: `--metal` and `--ir` → `tools/mslgen.cpp` → C++ compiled by MSVC `/O2`.
  - `simd`: the same, translated to AVX2 with 8 CUDA threads per register.
  Prebuilt mode commits `.metal` and `.ir` with SHA-256 files, bound to the source hash like the `.o` files, so builds without `kath` get all three paths.
- **`pcuda_mslgen`:**
  - Parses the Metal text for structure and the IR for semantics.
  - Fixes the two upstream Metal bugs: the sin/cos turns convention (F14) and lost byte signedness (F15).
  - Qualifies math calls through a `metal_stdlib` stand-in that reproduces the exact x86 operations (F16).
  - Fails the build on an IR operation it can't reproduce exactly.
- **SIMD translator:**
  - A recursive-descent parser for Booth's statement forms.
  - A fixed-point uniform/varying analysis: a value is varying if computed from varying values, or assigned under divergent control and observed outside that region.
  - Masked `if` and loop execution, with blends only where a partial-mask value escapes.
  - Varying pointers are kept as a uniform base plus a lane index.
  - Loads and stores touch only active lanes: masked loads/stores when contiguous, hardware gathers when strided, aligned-word gathers for bytes, and per-lane order for scattered stores.
  - Division and transcendentals run lane by lane, so faults and results match the scalar code.
  - Kernels it can't translate (helper calls, unsigned IR ops, non-canonical loops) fall back to `native`.
- **Runtime:**
  - `KernelEntry::native` points at the generated functions.
  - `pcudaSetKernelCodegen`, `pcudaGetKernelCodegenFor` and `PCUDA_CODEGEN` select the path. `auto` picks simd, then native, then booth, and checks AVX2 support via CPUID.
  - Native code runs under the same `pcuda_sysv_invoke` thunk through a small System V → Win64 bridge (`pcuda_sysv_native_bridge`). Fault containment, stack-overflow recovery and register checks are therefore unchanged. Faults report "native code" or "simd code" instead of a `.text` offset.
- **Pool:**
  - Chunked launches split into up to `workers × 4` chunks (`PCUDA_CHUNKS_PER_WORKER`).
  - Pool threads and the waiting caller spin 50 µs before sleeping (`PCUDA_POOL_SPIN_US`).
  - The caller runs chunks of its own launch.
  - A single-chunk launch runs inline.
  - Only as many sleepers are woken as there are extra chunks.
  - The chunk group is shared-owned by its jobs, which fixes a latent use-after-free when the caller returns as soon as the count reaches zero.
- **4-bit weights:** `matvec_q4`, using the ggml Q4_0 nibble layout with an f32 scale (5 bits per weight). For Llama 3.2 3B shapes that is 2.01 GB per token, against 2.02 GB for Ollama's Q4_K_M file.
- **Decode benchmark:**
  - Real autoregressive steps: K/V appended at each position, attention over `pos + 1` positions, the device `dims` rewritten per token.
  - Prompt length and token count are taken from Ollama.
  - The CUDA path copies the logits back to the host every token.
  - `scripts/compare_llm.ps1` runs Ollama (GPU and CPU-only) and the bench, and prints one table.

**Verification**
- `test_codegen`: every kernel produces bit-identical bytes on booth, native and simd, serial and chunked, at sizes 1, 7, 8, 9, 63, 1000, 4097 and partial row groups. Faults are contained on every path.
- The kernel-running suites also run as `<suite>_booth/_native/_simd`.
- Pass rates: 25/25 in the MSVC, prebuilt-only (fresh) and ASan (fresh) builds; 27/27 in the NVIDIA build.
- All pcuda paths give bit-identical logits for the 3B model.

**Results** (i5-13400F with 16 logical CPUs, RTX 4060; Llama 3.2 3B shapes, Q4; prompt 43, 128 tokens)

| Run | ms/token | tok/s | weights GB/s |
|---|---|---|---|
| pcuda booth path, 16 threads | 1,766 | 0.57 | 1.1 |
| pcuda native path, 16 threads | 216 | 4.6 | 9.3 |
| pcuda simd path, 16 threads | 156 | 6.4 | 12.9 |
| CUDA, same kernels | 65 | 15.4 | 30.9 |
| plain C++, 1 thread | 1,195 | 0.84 | 1.7 |
| Ollama llama.cpp, CPU only | 35 | 28.7 | 58 |
| Ollama llama.cpp, GPU | 9.3 | 107 | 216 |

Pool changes alone (`pcuda_bench`, vector_add):
- Single-chunk chunked launch overhead: 38 µs → 1.4 µs.
- 256K elements with 16 workers: 114 → 92 µs.
- 4M elements with 4 workers: 2.36 → 1.18 ms.

**Findings**
1. The Metal route recovers what Booth's x86 back end loses: 8× (native) and 11× (simd) per token on the same Booth IR, bit-identical.
2. The same source on the GPU is still 2.4× faster than simd. Real llama.cpp is another 4.5× faster on the CPU and 7× on the GPU. The remaining gap is algorithmic: llama.cpp uses integer dot products on quantized activations, and warp-cooperative, coalesced loads. Neither can be expressed in the one-thread-per-row kernel subset Booth's CPU backend accepts.
3. SIMD is limited by the per-row accumulation chain and strided gathers (8 rows per register), not by memory bandwidth (12.9 of about 60 GB/s).

**Next steps (not done):** interleave two 8-lane groups per call to hide add latency; carry kernels with `__shared__` and barriers (not supported by Booth's CPU backend).

## 22. Toward Ollama's CPU speed, step 1: 8-row interleaved weights (2026-11-06)

**Limits measured on this machine.**
- The i5-13400F has AVX2, FMA and AVX-VNNI, but no AVX-512.
- Memory is dual-channel DDR5-6000, which streams at 84 GB/s with 16 threads (73 GB/s with 10). At 2.02 GB per token that caps decode at about 41.6 tok/s.
- Ollama CPU-only runs at 28.4 tok/s (58 GB/s) with default threads, and at 26.4/29.5/28.6 tok/s with 6/10/16 threads.

**Change.**
- `matvec_q4x8` stores, for each group of 8 rows and each 32-weight block, byte j of all 8 rows side by side. Neighbouring threads therefore read neighbouring bytes. It is the same Booth-compatible one-thread-per-row CUDA C, with different index arithmetic.
- It sums each block in 4 independent partial sums, combined in a fixed order. That breaks the dependent-add chain that limited the SIMD path once the gathers were gone.
- The SIMD byte load gained a fast path: when all 8 lanes are active and contiguous, it does one 8-byte load widened to 8 lanes. Partial groups keep masked gathers.
- The adapter requires rows % 8 == 0, and chunks start on 8-row groups.
- `Weights::Q4X8` repacks the Q4 values (`ref::repackQ4x8`). It is now the default for `pcuda_llm_bench` and `compare_llm.ps1`.

**Results** (3B shapes, prompt 43, 128 tokens):

| ms/token | booth | native | simd | CUDA |
|---|---|---|---|---|
| q4 (row-major) | 1,795 | 208 | 182 | 65 |
| q4x8, layout only | 1,707 | 233 | 138 | 40 |
| q4x8 + 4 partial sums | 1,666 | 208 | **111** | **39** |

- SIMD is 1.64× faster than with row-major weights, at 9.0 tok/s (18 GB/s).
- The same kernels on CUDA reach 25.7 tok/s (51.5 GB/s), because their loads are now coalesced.
- All pcuda paths are bit-identical. Logits are within 3e-7 of the C++ reference.

**Remaining gap to Ollama's 28.4 tok/s** (about 3.2× on SIMD). The SIMD path is still compute-bound at about 1 µop per weight: per weight it converts to float, multiplies and adds, and a contiguity check runs on every load. Next steps:
- 2: 8-bit activations with integer dot products (AVX-VNNI `vpdpbusd`, through a `__dp4a`-style helper that the generator maps to VNNI).
- 3: fewer launches and barriers per token.

## 23. Step 2: 8-bit activations and integer dot products (AVX-VNNI) (2026-11-06)

**Change.**
- **`quantize_q8x4`:** per block of 32 activations, d = max|x|/127 and q = round(x/d), half away from zero, clamped to ±127. It writes packed int8 words plus {d, Σq}.
- **`matvec_q4x8_q8`:** 4-bit weights in an 8-row interleaved int stream with the f32 block scales inside it. That's the same 5 bits per weight, and it keeps the kernel within Booth's 5-parameter limit through a pointer cast. Per block it computes dot = Σ nibble·q with two independent dp4a chains, then (dot − 8·Σq) · d_w · d_x.
- **`pcuda_dp4a`:** a `__device__` helper doing a u8 × s8 dot product of 4 packed bytes.
  - On nvcc it is `__dp4a`, selected by `__NVCC__`.
  - Elsewhere it is generic byte arithmetic, which is what Booth compiles.
  - `pcuda_mslgen` maps calls to it onto AVX-VNNI `_mm256_dpbusd_avx_epi32`, after checking that the helper's Metal body matches the canonical body exactly. CPUs without VNNI get an exact AVX2 fallback.
- **Translator support:** `as_type` pointer casts (`ptrcast`), verified intrinsic helpers, and signed rewriting of literals Booth prints as unsigned (F17).
- **Model and benchmark:** `Weights::Q4X8Q8` (`ref::repackQ4x8Q8`), with per-backend activation scratch (`Backend::setScratch`). It is the default in `pcuda_llm_bench` and `compare_llm.ps1`.
- **Streams:** the stream worker and `runSync` callers now spin 50 µs before sleeping (`PCUDA_POOL_SPIN_US`). Before, every synchronous launch paid a thread wake-up, which mattered for the roughly 300 tiny launches per token.
- **Two more Booth Metal bugs found:** F17 (negative constants printed as unsigned) and F18 (phi copy dropped for an `if` without `else`). The kernels avoid the F18 pattern, and MSVC C4701 stays an error to catch it.

**Verification.**
- `quantize_q8x4` and `matvec_q4x8_q8` are bit-exact against their C++ references, chunked too, and bit-identical on booth, native and simd.
- `matvec_q4x8_q8` is within 3e-2 of the float `matvec_q4` (activation rounding).
- Pass rates: 27/27 NVIDIA; 25/25 normal, prebuilt-only and ASan.

**Results** (3B shapes, prompt 43, 128 tokens):

| ms/token | booth | native | simd | CUDA |
|---|---|---|---|---|
| q4x8 (float activations) | 1,666 | 208 | 111 | 39 |
| q4x8q8 (8-bit activations, dp4a / VNNI) | 1,004 | 229 | 63 | 33 |
| + stream spin | 995 | 218 | **51** | 33 |

- The SIMD path reaches 18–19.5 tok/s (36–39 GB/s), up from 6.4 tok/s at the start of §21.
- Ollama CPU-only measured 28.2 tok/s in the same run.
- The same kernels on CUDA reach 30 tok/s.

**Remaining gap to Ollama CPU (about 1.5×).**
- Matrix-vector work is about 41 ms per token, roughly 49 GB/s within matvec. The rest (about 8 ms) is about 300 small launches: rms_norm, rope, attention, silu, add and quantize.
- Next steps (step 3):
  - fuse q/k/v and gate/up into single launches and quantize once per input;
  - run rms_norm as row-parallel chunks;
  - record a token as one pool job with barriers instead of about 470 separate launches.

## 24. Step 3: fewer launches per token (2026-11-06)

**Change.**
- **Fused projections:** q/k/v and gate/up are stored as one matrix each, with their rows concatenated. Each is one launch with one activation quantization, writing into contiguous `q|k|v` and `g|u` buffers.
- **Fused kernels:**
  - `add_rms_norm` performs the residual add and the following RMS norm, including across the layer boundary into the next layer's attention norm or the final norm.
  - `silu_mul_q8x4` performs SwiGLU and 8-bit quantization, so `h` is never stored.
  Both perform the same float operations in the same order as the sequences they replace, so they are bit-identical to them (tested).
- **Launch counts:** about 15 per layer instead of 24, plus the two KV-append copies.
- **Small launches:** a launch with fewer than 8 threads takes the native path instead of SIMD. Most lanes would be idle while still paying for masked loads. The results are bit-identical.
- **Bit-exact C++ reference:** its `rope` now reproduces Booth's sin/cos lowering (F11), so it matches every pcuda path bit for bit. Before, an 8-bit activation rounding could flip and move the logits by 1e-4.
- **`compare_llm.ps1` fairness:** it now waits for Ollama to unload the model and then pauses 15 s before the bench. Running straight after Ollama cost the bench about 15%.

**Results** (3B shapes, prompt 43, 128 tokens):

| | ms/token | tok/s |
|---|---|---|
| simd, step 2 | 51–56 | 18–19.5 |
| simd, step 3 (standalone) | 45.6 | 21.9 |
| simd, step 3 (compare_llm.ps1, after cooldown) | 49.4 | 20.2 |
| Ollama llama.cpp, CPU only (same run) | 35.4 | 28.25 |
| CUDA, same kernels | 33.4 | 29.9 |

Per-token profile at the first position:

| Component | Time |
|---|---|
| Matrix-vector | about 33 ms (about 61 GB/s within matvec) |
| Attention | 3 ms at position 43, 7.7 ms at position 167 |
| rope, KV append, silu, add+norm | about 3 ms |
| Overhead between launches | about 2.5 ms |

**Remaining gap to Ollama CPU (about 1.3–1.4×).** Next candidates:
1. **Attention data layout:** store K and V per head with positions contiguous, so the 8 SIMD lanes read adjacent floats. Today `attn_scores` gathers rows that are 1024 floats apart. Saves about 2–4 ms at longer contexts.
2. **Lane-uniform analysis in `pcuda_mslgen`:** prove at translation time that values like `row / 8` are the same in every lane and `row % 8` equals the lane number. Loads such as `qw[base + w*8 + l]` then become plain contiguous loads with no runtime contiguity check. This is the "warp-uniform" analysis GPU compilers do. Estimated 10–20% on matvec.
3. **Run a token as one pool job** with barriers between kernels, removing the remaining launch overhead.

## 25. Steps 4–6: lane-affine translation, transposed K, pool sizing (2026-11-06)

**Measuring.** `pcuda_llm_bench --paths simd,cuda` times only the chosen paths. Timing the slow booth and native paths first heats the CPU, and that slowed simd by up to 15% (59.6 vs 50.7 ms for the same build). `compare_llm.ps1` now passes `--paths simd,cuda` by default.

**Step 4: lane-affine values in `pcuda_mslgen`** (candidate 2 from §24).
- The analysis gives each value one of three shapes:
  - **Uni**: the same in every lane;
  - **Aff**: lane `l` holds `b + s*l`, with `b` scalar and `s` a constant;
  - **Gen**: a full vector.
- It is an optimistic fixed point; shapes only move down.
- Uni and Aff values are computed by the scalar C expression on lane 0's values, so they cost nothing per lane.
- An `aligned` flag records a non-negative multiple of 8. `tid` is aligned because groups start at multiples of 8, and `blockIdx * blockDim` folds to 0.
- For an aligned Aff value with stride 1 and an aligned divisor `D`:
  - `x / D` is Uni;
  - `x % D` is Aff with stride 1.
- Loads and stores through Aff pointers with stride 1 use `loadc`/`storec`: masked contiguous accesses with no run-time index checks.
- A vector pointer derived from an Aff pointer demotes it to Gen, because blends keep one scalar base.

The `matvec_q4x8_q8` inner loop became scalar address math, one `vpmaskmovd` and two `vpdpbusd`. Result: 50.7 → 43.7 ms.

**VNNI choice per launch.** `vdp4a` checked `cpuHasAvxVnni()` (a thread-safe static) at every call. The possible fallback call made MSVC spill all live `ymm` registers around each dot product.
- The generated `group` is now `template <bool kVnni_>`, and the entry point picks the instance once.
- The inner loop is 12 instructions with no spills.
- Result: 43.0 → 41.2 ms.

**Pool dispatch.** Workers claim chunks with an atomic counter on the launch's group, with no lock per chunk. One `WakeByAddressAll` per launch wakes sleepers, replacing up to 15 `notify_one` calls. The caller waits on the remaining-count with `WaitOnAddress`. Result: about 1 ms.

**Step 5: transposed K and `pcuda_aligned8`** (candidate 1 from §24).
- **`pcuda_aligned8(x)`:** a `__device__` identity helper that promises `x >= 0` and `x % 8 == 0`. `pcuda_mslgen` recognizes its exact Booth body (`return p0;`), marks uniform results aligned, and emits `pcuda::simd::aligned8()`. That check faults with `__debugbreak` (contained like any kernel fault) if the promise is broken.
- **Layouts and dims:**
  - The K cache is transposed: `kt[kv_stride][k_stride]`.
  - Score rows are padded to `ctx_pad`, a multiple of 8.
  - The attention dims grow to 7 ints: `ctx_pad` and `k_stride` are added.
- **Generated code:**
  - In `attn_scores`, `h` is Uni, `t` is Aff, `q` is a scalar broadcast, and K is one contiguous load.
  - In `attn_values`, `h` is Uni and `d` is Aff, so `p` is a scalar broadcast and V is contiguous.
- **`kv_append`:** a new kernel that writes K transposed and V in one launch. It replaces two copies. Booth allows at most 5 integer/pointer parameters, so it reads `pos = ctx - 1` from the attention dims.
- CUDA runs the same kernels. Its result moved closer to the reference (1.9e-4 → 5.5e-7), with no change in speed.
- Result: attention 3.55 → 2.8 ms at position 43; 41.2 → 40.1 ms per token.
- Chunk boundaries are now multiples of 16 threads, down from 64. Kernels with few threads, such as `silu_mul_q8x4`'s 256, now split over all workers.

**Step 6: pool size.** Longer spinning (200–5000 µs) made decode up to 5× slower, even with `UMWAIT`.
- **Cause:** oversubscription. Launches are synchronous: the host thread waits while the stream thread runs the launch, with 16 pool threads alongside. That is 18 threads on 16 logical processors, so waiting threads took processors from the threads with work.
- **Sweep:** pool sizes of 11–12 were fastest. This CPU has 12 performance-core logical processors; busy efficiency cores also slow the shared ring and memory path, and llama.cpp avoids them too.
- **Default:** P-core logical processors − 1 (11 here), from `GetLogicalProcessorInformationEx` `EfficiencyClass`. The launching thread runs chunks too.
- **Waiting:** `UMWAIT` (C0.1) replaces `pause` spinning when the CPU has WAITPKG. The 50 µs default stays.

**Results** (`compare_llm.ps1`, two consecutive runs):

| | ms/token | tok/s |
|---|---|---|
| simd, step 3 | 46–49 | 20–22 |
| simd, steps 4–6 | 36.4–36.9 | 27.1–27.5 |
| Ollama llama.cpp, CPU only (same runs) | 38.2 / 33.9 | 26.2 / 29.5 |
| CUDA, same kernels | 33.2 | 30.1 |

pcuda's simd path from the same CUDA C sources now matches Ollama on CPU within run-to-run noise. All CPU paths remain bit-identical, and also bit-identical to the C++ reference.

**Remaining time** at position 43:
- matvec 30–31 ms: lm_head runs at about 80 GB/s, FFN at about 70 GB/s, and the small attention projections at about 55 GB/s;
- attention 1.8 ms;
- rope + KV append 1.6 ms;
- silu 1.0 ms;
- add+norm 0.5 ms.

**Next candidates:**
- run a token as one pool job (§24 candidate 3);
- fuse rope into `kv_append`;
- vectorize `exp2`/`sin`/`cos` with one shared deterministic implementation for all paths.

## 26. Step 7: each token as one graph (2026-11-06)

**API** (like CUDA graphs): `pcudaStreamBeginCapture`, `pcudaStreamEndCapture`, `pcudaGraphLaunch` and `pcudaGraphDestroy`.
- **During a capture:** launches on the stream are validated and marshalled at once, then recorded as nodes, each with one argument set per chunk. They return Success without running.
- **Refused during a capture** (`UnsupportedOperation`, as CUDA does), because they would run ahead of the recorded work:
  - copies, memsets, frees and device synchronization while any stream captures;
  - stream synchronization, host functions and event records on the capturing stream itself.
- **Graph data:** immutable. The per-chunk tasks are built once at `EndCapture`, and a queued launch keeps the data alive after `pcudaGraphDestroy`. `pcudaDeviceReset` drops all graphs and captures.

**Execution.** `WorkerPool::runSteps` runs a list of steps, one step per kernel, as one job.
- Threads claim the current step's chunks with an atomic counter.
- The thread finishing a step's last chunk opens the next step with one release store plus `WakeByAddressAll`. Threads with nothing left to claim wait in `UMWAIT`, then `WaitOnAddress`.
- A failure ends the job after its step. The reported error is the first in (step, chunk) order, with the node index and kernel name.
- An ordinary chunked launch is now the one-step case of the same code.

**Decode.** `Backend::beginToken`/`endToken` bracket a token.
- The pcuda backend captures the token's ~450 launches and runs them as one synchronous graph launch.
- The bench's CUDA backend captures a CUDA graph, updates the one executable graph in place (`cudaGraphExecUpdate`), and launches it, as llama.cpp does.
- Per-op profiling still runs launches one by one.
- `pcuda_llm_bench --graphs on|off` (default on).

**Results:**
- **Graphs on vs off**, alternating runs: 33.3/34.0 vs 37.5/37.7 ms, about 4 ms saved per token.
- **`compare_llm.ps1`, two runs:**
  - pcuda simd 34.0 and 33.1 ms (29.4 and 30.2 tok/s);
  - Ollama CPU-only 35.0 and 34.8 ms (28.6 and 28.7 tok/s);
  - CUDA with the same kernels and CUDA graphs: 33.1 ms (30.2 tok/s), with no gain from graphs, because its time is in the naive kernels.

**Correctness:** all CPU paths remain bit-identical to each other and to the C++ reference with graphs on. New tests:
- `test_graph` on every code path: order and barriers, async launch with destroy, a capture on one stream leaving the default stream running, capture rules, and a fault naming its node with later nodes skipped;
- graph decode bit-identical to direct launches.

Stability: 150 repeated runs of the pool and stream tests, plus AddressSanitizer.

## 27. Step 8: prefetch, parallel norm + quantize, one rope launch (2026-11-06)

**Software prefetch.** `pcuda_mslgen` emits `prefetchAhead(p)` before every stride-1 affine load inside a loop: an `_mm_prefetch` some distance past the address.
- **Why:** the hardware streamer stops at 4 KB page boundaries, and each thread streams its weight rows sequentially.
- **Sweep:** 0 / 256 / 512 / 1 K / 1.5 K / 2 K / 3 K / 4 K / 8 K bytes ahead. 1.5–8 KB all gave 32.2–32.9 ms, against 33.6–33.8 ms with prefetch off.
- **Default:** 2 KB, with `PCUDA_PREFETCH_BYTES` to override. A prefetch never faults, so reading ahead past a buffer's end is safe.

**Parallel norm + quantize.**
- **Before:** `add_rms_norm` ran on one thread, followed by a separate `quantize_q8x4` launch.
- **Now:** two chunkable kernels.
  - `add_sumsq`: one thread per block of 32 does `x += t` and writes the block's sum of squares.
  - `norm_q8x4`: every thread adds the per-block sums in the same order, then normalizes its block and quantizes it into the 8-bit scratch.
- **Scratch layout:** Booth's 5-parameter limit led to one int buffer per call: `[nb*8 words | nb*2 scales | nb partial sums]`. Every producer and consumer finds the scales right after the words for its `nb` (`llm::scratchInts`, `llm::scratchScales`).
- **Matrix inputs:** the qkv, gate/up and lm_head matvecs read the scratch directly. Only the input of the attention-output matvec is still quantized separately, now in chunks.
- **Summation order:** the sum of squares is now added per block, then across blocks. The C++ reference (`ref::addSumsq`, `ref::normQ8x4`) uses the same order, so all CPU paths stay bit-identical to it. Against the old order, the quantized bytes differ only on rare rounding ties (tested: at least 95% identical).

**Rope:** q and k are adjacent in the qkv buffer and use the same formula, so one launch covers `heads + kv_heads` heads.

**Results:**
- **Repeated runs:** simd 31.8–33.2 ms (30.1–31.4 tok/s). The prefetch gives most of it. The parallel norm saves at most about 0.5 ms for pcuda, because graphs had already made the serial steps cheap.
- **CUDA, same kernels:** 33 → 20.4 ms. Its one-thread `add_rms_norm` had cost 13 ms per token.
- **`compare_llm.ps1`:**

| | ms/token | tok/s |
|---|---|---|
| pcuda simd | 32.5 | 30.7 |
| Ollama CPU | 35.3 | 28.3 |
| CUDA, same kernels | 20.4 | 49.0 |
| Ollama GPU | 9.3 | 107.7 |
