; System V -> Win64 shims for the libm functions Booth kernels may call.
;
; Booth calls externals System V style after `and rsp, -16`, so on entry rsp = 8 mod 16
; and there is no 32-byte shadow space. A Win64 callee may write that shadow space,
; which would land inside Booth's stack frame. Each shim allocates shadow space,
; realigns, and calls a C++ wrapper. Float argument and result use xmm0 in both ABIs,
; and Win64 callees preserve a superset of what System V callers expect.

OPTION CASEMAP:NONE

EXTERN pcuda_crt_sinf:PROC
EXTERN pcuda_crt_cosf:PROC
EXTERN pcuda_crt_exp2f:PROC
EXTERN pcuda_crt_log2f:PROC

PUBLIC pcuda_sysv_sinf
PUBLIC pcuda_sysv_cosf
PUBLIC pcuda_sysv_exp2f
PUBLIC pcuda_sysv_log2f
PUBLIC pcuda_sysv_native_bridge

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

SYSV_SHIM pcuda_sysv_sinf,  pcuda_crt_sinf
SYSV_SHIM pcuda_sysv_cosf,  pcuda_crt_cosf
SYSV_SHIM pcuda_sysv_exp2f, pcuda_crt_exp2f
SYSV_SHIM pcuda_sysv_log2f, pcuda_crt_log2f

; Runs a native (Win64) kernel function under pcuda_sysv_invoke, so it gets the same fault
; containment and register checks as Booth code. System V arguments: rdi = const LaunchArgs*,
; rsi = void (*)(const LaunchArgs*). Win64 callees preserve every register System V callers
; rely on (and rdi, rsi, xmm6-15 too); this frame supplies the 32-byte shadow space and keeps
; rsp 16-byte aligned at the call.
pcuda_sysv_native_bridge PROC FRAME
        sub     rsp, 40
        .allocstack 40
        .endprolog
        cld
        mov     rcx, rdi
        call    rsi
        add     rsp, 40
        ret
pcuda_sysv_native_bridge ENDP

_TEXT ENDS
END
