; System V -> Win64 shims for the libm functions Booth kernels may call.
;
; Booth calls externals System V style after `and rsp, -16`, so on entry rsp = 8 mod 16
; and there is no 32-byte shadow space. A Win64 callee may write that shadow space,
; which would land inside Booth's stack frame. Each shim allocates shadow space,
; realigns, and calls a C++ wrapper. Float argument and result use xmm0 in both ABIs,
; and Win64 callees preserve a superset of what System V callers expect.

OPTION CASEMAP:NONE

EXTERN kc_crt_sinf:PROC
EXTERN kc_crt_cosf:PROC
EXTERN kc_crt_exp2f:PROC
EXTERN kc_crt_log2f:PROC

PUBLIC kc_sysv_sinf
PUBLIC kc_sysv_cosf
PUBLIC kc_sysv_exp2f
PUBLIC kc_sysv_log2f
PUBLIC kc_sysv_native_bridge

SYSV_SHIM MACRO shim_name, target
shim_name PROC FRAME
        sub     rsp, 40
        .allocstack 40
        .endprolog
        cld
        call    target
        add     rsp, 40
        ret
shim_name ENDP
ENDM

_TEXT SEGMENT

SYSV_SHIM kc_sysv_sinf,  kc_crt_sinf
SYSV_SHIM kc_sysv_cosf,  kc_crt_cosf
SYSV_SHIM kc_sysv_exp2f, kc_crt_exp2f
SYSV_SHIM kc_sysv_log2f, kc_crt_log2f

; Runs a native (Win64) kernel function under kc_sysv_invoke, so it gets the same fault
; containment and register checks as Booth code. System V arguments: rdi = const LaunchArgs*,
; rsi = void (*)(const LaunchArgs*). Win64 callees preserve every register System V callers
; rely on (and rdi, rsi, xmm6-15 too); this frame supplies the 32-byte shadow space and keeps
; rsp 16-byte aligned at the call.
kc_sysv_native_bridge PROC FRAME
        sub     rsp, 40
        .allocstack 40
        .endprolog
        cld
        mov     rcx, rdi
        call    rsi
        add     rsp, 40
        ret
kc_sysv_native_bridge ENDP

_TEXT ENDS
END
