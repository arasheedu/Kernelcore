# Review of `cuda_cpu_poc_plan.v1-linux.md`

**Reviewed:** 2026-11-06  
**Outcome:** The architecture and scoping are sound. The plan is not executable on native Windows as written, and several of its details would cause silent wrong results or crashes even on Linux. `cuda_cpu_poc_plan.md` has been rewritten for Windows with the fixes below.

Facts below marked **[verified]** were checked against Booth's source at `master` (`src/cpu/cpu_elf.c`, `src/cpu/cpu_emit.c`, `examples/cpu_launch_vadd.c`) on 2026-11-06. Re-check them against the pinned commit in Phase 0.

---

## 1. What the plan gets right

- Narrow scope: one kernel, typed adapters, no claims of CUDA ABI compatibility.
- Clean backend seam (`IBackend`) with an NVIDIA backend deferred.
- Runtime-owned allocation registry, explicit errors, honest device naming.
- Treats the N-core chunking as an opt-in experiment, not CUDA semantics.
- Pins the upstream version and puts the ABI in one translation unit.

## 2. Blocking problems for a Windows target

| # | Problem | Why it blocks Windows |
|---|---|---|
| W1 | The plan's only answer to "Windows" is "use WSL2". Native Windows is listed as a non-goal. | Doesn't meet the goal. |
| W2 | **[verified]** Booth's `--cpu` writes **ELF64 relocatable objects only**. There is no COFF/PE path. | MSVC's `link.exe` and `lld-link` can't link it. |
| W3 | **[verified]** Generated code uses the **System V AMD64 calling convention** (args in `rdi, rsi, rdx, rcx, r8, r9`, `xmm0–7`) and does not preserve `rsi`, `rdi` or `xmm6–15`. | Even after converting ELF to COFF, calling it from MSVC/Win64 code corrupts the caller's registers and passes arguments in the wrong registers. Converting with `objcopy` doesn't fix the ABI. |
| W4 | **[verified]** Kernels may call `sinf`, `cosf`, `exp2f`, `log2f` as undefined externals, using a System V call with no 32-byte shadow space. | Binding these straight to the Windows CRT lets the CRT write its shadow space into Booth's stack frame and corrupt it. They need System V-to-Win64 shims. |
| W5 | Uses `std::aligned_alloc`. | MSVC's CRT doesn't provide it. |
| W6 | Phase 5 exit criteria require ThreadSanitizer. | TSan isn't available for MSVC or clang-cl on Windows. |
| W7 | The Phase 0 smoke test is `gcc -no-pie examples/cpu_launch_vadd.c vadd.o`. | Can't run on Windows. |
| W8 | Worker threads use the default stack size. | The Windows default is 1 MB versus 8 MB on Linux, and Booth's codegen is "stack-everything" with no register allocator. Large kernels risk overflowing the stack. |

## 3. Correctness bugs (any platform)

| # | Bug | Consequence |
|---|---|---|
| C1 | **[verified]** The adapter declares `int nthreads`. Booth stores the full 64-bit register and compares it with a 64-bit `cmp`. | The upper 32 bits are undefined after an `int` argument. The thread loop can run billions of iterations and write far out of bounds. **Pass every integer argument as a sign-extended 64-bit value.** |
| C2 | **[verified]** Booth emits only the **first** `__global__` function per `.cu` file. | A file with two kernels silently loses the second. The plan must require one kernel per file. |
| C3 | **[verified]** `blockIdx` is constant 0, `gridDim` is 1, and `blockDim` equals `nthreads`. | Idiomatic `blockIdx.x*blockDim.x+threadIdx.x` works only for a single call. This needs to be documented as part of the kernel contract. |
| C4 | Chunking (§11) offsets pointers and sets `nthreads = chunk_length` but leaves `count` unchanged. | For the `if (i < count)` kernel this is harmless, but any kernel that uses `count` for indexing or striding would read out of range. Each chunk must also get `count = chunk_length`, or the kernel needs an explicit offset parameter. |
| C5 | **[verified]** Booth only loads register arguments for calls it makes (6 GPRs). | A kernel with more than 5 integer/pointer parameters (5 plus the hidden `nthreads`) isn't supported reliably. Reject it at build time. |
| C6 | `count` is `int` while buffer sizes are `size_t`. | Narrowing above `INT_MAX` without any check. |
| C7 | `KernelEntry::launch_cpu` returns `void`. | Contradicts design rule 5: a launch can't report failure. |
| C8 | `DeviceToDevice` copy semantics for overlapping ranges are unspecified. Zero-byte `malloc`/`memcpy` behaviour is unspecified. | Undefined or inconsistent behaviour. |

## 4. API and design issues

- Inconsistent naming: §2 says `kcMalloc`, while §6 declares `kc::malloc`, `kc::free` and `kc::memcpy`. The namespace versions collide with CRT macros (`_CRTDBG_MAP_ALLOC` redefines `malloc`/`free`) and invite ADL and `using namespace` accidents. Use the `kc*` names.
- The error model has no sticky "context poisoned" state. CUDA treats an illegal address as sticky. The POC needs the same, or a faulted kernel's partial writes go unnoticed.
- No `NotInitialized`, `InvalidKernelImage`, `IllegalAddress`, `KernelFault` or `DynamicCodeBlocked` errors.
- No shutdown story. Worker threads owned by statics and joined in static destructors can deadlock under the Windows loader lock, especially in a DLL. Use explicit `kcShutdown()`.
- Nothing contains faults: a bad kernel crashes the whole process with no diagnostic.

## 5. Process and plan-quality issues

- **Phase 0 has no go/no-go gate and no fallback.** If `kath` doesn't build, or the ABI isn't what we assumed, the plan has nowhere to go.
- Getting `kath` relies on a single path. Booth's release notes mention Windows binaries with `SHA256SUMS`, but asset names weren't confirmed. There's no plan for checksum checks, building from source on Windows, or running `kath` without it installed.
- §13 compiles the kernel by hand and then runs CMake, so the same step happens twice in different ways. The build should be the single source of truth.
- The build doesn't validate the generated object (magic number, machine type, symbol, relocation types). "Booth returned 0" isn't enough.
- Benchmarks don't say what clock, affinity, power plan or warm-up they use.
- CI is left unspecified, and so is how tests run without network access (no test framework dependency is pinned).
- Line endings aren't handled. A CRLF checkout of `.cu` files on Windows is an untested input to `kath`.

## 6. Summary of changes made in the revised plan

1. **Native Windows x64 is the primary target.** WSL is now only an optional fallback for building `kath`, plus an optional TSan job.
2. **ELF loader in the runtime:** a small, bounds-checked loader for Booth's restricted ELF subset (one `.text`, one relocation type `R_X86_64_PLT32`, allowlisted externals). It maps the code with `VirtualAlloc` and switches it from writable to executable.
3. **MASM System V invoke thunk (`ml64`):** saves all Win64 non-volatile registers including `xmm6–15`, marshals arguments as 64-bit values, and provides a recovery point after a fault.
4. **System V-to-Win64 shims** for `sinf`, `cosf`, `exp2f` and `log2f`, placed next to the loaded code so `rel32` always reaches them.
5. **Fault containment:** a vectored exception handler turns access violations and divide-by-zero inside kernel code into `KernelFault`, with a sticky poisoned context. Stack overflow fails fast with a diagnostic.
6. **Guard-page allocator** (`VirtualAlloc` with a trailing `PAGE_NOACCESS` page) so out-of-bounds writes fault at the first bad byte.
7. **Four ways to get the kernel object**, chosen with `KC_KATH_MODE=auto|exe|wsl|prebuilt`: pinned `kath.exe` with SHA-256 → source build → `kath` inside WSL → checked-in prebuilt `.o` with SHA-256. The build doesn't depend on any one of them.
8. **The object is embedded as a byte array** in the executable, so nothing is loaded from a path at runtime.
9. Build-time object validation (`kc_objcheck`), a hard Phase 0 go/no-go gate, all integer arguments passed as 64-bit, one kernel per file, an argument-count limit, explicit shutdown, an 8 MB worker stack, ASan instead of TSan, and a Windows CI matrix.
