; Win64 -> System V AMD64 call thunk for Booth --cpu kernels.
;
; int kc_sysv_invoke(const void* entry,      ; rcx
;                       const uint64_t* gpr6,   ; rdx  -> rdi, rsi, rdx, rcx, r8, r9
;                       const uint64_t* xmm8,   ; r8   -> xmm0..xmm7 (low 64 bits)
;                       void** anchor_slot)     ; r9   receives the post-prologue rsp
; Returns 0 when the kernel returns normally, 1 when resumed at kc_sysv_recover
; by the fault guard after a hardware exception inside the kernel.
;
; Saves every Win64 non-volatile register (rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15),
; MXCSR and the x87 control word, because Booth's System V code does not preserve
; rdi/rsi/xmm6-15. The stack is 16-byte aligned at the call, as System V requires.

OPTION CASEMAP:NONE

PUBLIC kc_sysv_invoke
PUBLIC kc_sysv_recover

_TEXT SEGMENT

kc_sysv_invoke PROC FRAME
        push    rbp
        .pushreg rbp
        push    rbx
        .pushreg rbx
        push    rdi
        .pushreg rdi
        push    rsi
        .pushreg rsi
        push    r12
        .pushreg r12
        push    r13
        .pushreg r13
        push    r14
        .pushreg r14
        push    r15
        .pushreg r15
        ; entry rsp = 8 mod 16; 8 pushes -> 8 mod 16; 168 more -> 0 mod 16
        sub     rsp, 168
        .allocstack 168
        movaps  xmmword ptr [rsp+0],   xmm6
        .savexmm128 xmm6, 0
        movaps  xmmword ptr [rsp+16],  xmm7
        .savexmm128 xmm7, 16
        movaps  xmmword ptr [rsp+32],  xmm8
        .savexmm128 xmm8, 32
        movaps  xmmword ptr [rsp+48],  xmm9
        .savexmm128 xmm9, 48
        movaps  xmmword ptr [rsp+64],  xmm10
        .savexmm128 xmm10, 64
        movaps  xmmword ptr [rsp+80],  xmm11
        .savexmm128 xmm11, 80
        movaps  xmmword ptr [rsp+96],  xmm12
        .savexmm128 xmm12, 96
        movaps  xmmword ptr [rsp+112], xmm13
        .savexmm128 xmm13, 112
        movaps  xmmword ptr [rsp+128], xmm14
        .savexmm128 xmm14, 128
        movaps  xmmword ptr [rsp+144], xmm15
        .savexmm128 xmm15, 144
        .endprolog

        stmxcsr dword ptr [rsp+160]
        fnstcw  word ptr [rsp+164]
        mov     qword ptr [r9], rsp         ; recovery anchor

        mov     rax, rcx                    ; entry
        mov     r10, rdx                    ; gpr6
        mov     r11, r8                     ; xmm8
        movq    xmm0, qword ptr [r11+0]
        movq    xmm1, qword ptr [r11+8]
        movq    xmm2, qword ptr [r11+16]
        movq    xmm3, qword ptr [r11+24]
        movq    xmm4, qword ptr [r11+32]
        movq    xmm5, qword ptr [r11+40]
        movq    xmm6, qword ptr [r11+48]
        movq    xmm7, qword ptr [r11+56]
        mov     rdi, qword ptr [r10+0]
        mov     rsi, qword ptr [r10+8]
        mov     rdx, qword ptr [r10+16]
        mov     rcx, qword ptr [r10+24]
        mov     r8,  qword ptr [r10+32]
        mov     r9,  qword ptr [r10+40]
        cld
        call    rax
        xor     eax, eax

restore_state:
        cld
        ldmxcsr dword ptr [rsp+160]
        fninit
        fldcw   word ptr [rsp+164]
        movaps  xmm6,  xmmword ptr [rsp+0]
        movaps  xmm7,  xmmword ptr [rsp+16]
        movaps  xmm8,  xmmword ptr [rsp+32]
        movaps  xmm9,  xmmword ptr [rsp+48]
        movaps  xmm10, xmmword ptr [rsp+64]
        movaps  xmm11, xmmword ptr [rsp+80]
        movaps  xmm12, xmmword ptr [rsp+96]
        movaps  xmm13, xmmword ptr [rsp+112]
        movaps  xmm14, xmmword ptr [rsp+128]
        movaps  xmm15, xmmword ptr [rsp+144]
        add     rsp, 168
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rsi
        pop     rdi
        pop     rbx
        pop     rbp
        ret

        ; The fault guard resumes here with rsp = anchor (the post-prologue rsp).
kc_sysv_recover::
        mov     eax, 1
        jmp     restore_state
kc_sysv_invoke ENDP

_TEXT ENDS
END
