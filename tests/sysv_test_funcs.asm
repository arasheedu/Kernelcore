; Test helpers for the Win64 -> System V thunk.
;
; sysv_* functions are System V-convention "kernels" that misbehave on purpose.
; pcuda_test_check_preserved is a Win64 harness: it loads sentinels into every Win64
; non-volatile register, calls pcuda_sysv_invoke, and reports which ones changed.

OPTION CASEMAP:NONE

EXTERN pcuda_sysv_invoke:PROC

PUBLIC pcuda_test_check_preserved
PUBLIC sysv_echo_clobber
PUBLIC sysv_fault_read
PUBLIC sysv_div0
PUBLIC sysv_recurse

_DATA SEGMENT ALIGN(16) 'DATA'
ALIGN 16
xmm_pattern QWORD 0606060606060601h, 0606060606060602h
            QWORD 0707070707070701h, 0707070707070702h
            QWORD 0808080808080801h, 0808080808080802h
            QWORD 0909090909090901h, 0909090909090902h
            QWORD 0A0A0A0A0A0A0A01h, 0A0A0A0A0A0A0A02h
            QWORD 0B0B0B0B0B0B0B01h, 0B0B0B0B0B0B0B02h
            QWORD 0C0C0C0C0C0C0C01h, 0C0C0C0C0C0C0C02h
            QWORD 0D0D0D0D0D0D0D01h, 0D0D0D0D0D0D0D02h
            QWORD 0E0E0E0E0E0E0E01h, 0E0E0E0E0E0E0E02h
            QWORD 0F0F0F0F0F0F0F01h, 0F0F0F0F0F0F0F02h
clobber_mxcsr DWORD 7F80h      ; round toward zero, all exceptions masked
_DATA ENDS

_TEXT SEGMENT

CHECK_GPR MACRO reg, value, bit
        mov     r11, value
        cmp     reg, r11
        je      @F
        or      r10, 1 SHL bit
@@:
ENDM

CHECK_XMM MACRO reg, index
        movdqa  xmm0, xmmword ptr [rax + 16 * index]
        pcmpeqb xmm0, reg
        pmovmskb ecx, xmm0
        cmp     ecx, 0FFFFh
        je      @F
        or      r10, 1 SHL (8 + index)
@@:
ENDM

; uint64_t pcuda_test_check_preserved(entry, gpr6, xmm8, anchor_slot)
; returns (corruption_mask << 8) | thunk_result
;   mask bits: 0 rbx, 1 rbp, 2 rdi, 3 rsi, 4-7 r12-r15, 8-17 xmm6-xmm15, 18 DF set, 19 MXCSR changed
pcuda_test_check_preserved PROC FRAME
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
        sub     rsp, 216                ; 32 shadow + 160 xmm saves + 24 scratch
        .allocstack 216
        movaps  xmmword ptr [rsp+32],  xmm6
        .savexmm128 xmm6, 32
        movaps  xmmword ptr [rsp+48],  xmm7
        .savexmm128 xmm7, 48
        movaps  xmmword ptr [rsp+64],  xmm8
        .savexmm128 xmm8, 64
        movaps  xmmword ptr [rsp+80],  xmm9
        .savexmm128 xmm9, 80
        movaps  xmmword ptr [rsp+96],  xmm10
        .savexmm128 xmm10, 96
        movaps  xmmword ptr [rsp+112], xmm11
        .savexmm128 xmm11, 112
        movaps  xmmword ptr [rsp+128], xmm12
        .savexmm128 xmm12, 128
        movaps  xmmword ptr [rsp+144], xmm13
        .savexmm128 xmm13, 144
        movaps  xmmword ptr [rsp+160], xmm14
        .savexmm128 xmm14, 160
        movaps  xmmword ptr [rsp+176], xmm15
        .savexmm128 xmm15, 176
        .endprolog

        mov     rbx, 0B0B0B0B0B0B0B0B0h
        mov     rbp, 0B1B1B1B1B1B1B1B1h
        mov     rdi, 0B2B2B2B2B2B2B2B2h
        mov     rsi, 0B3B3B3B3B3B3B3B3h
        mov     r12, 0B4B4B4B4B4B4B4B4h
        mov     r13, 0B5B5B5B5B5B5B5B5h
        mov     r14, 0B6B6B6B6B6B6B6B6h
        mov     r15, 0B7B7B7B7B7B7B7B7h
        lea     rax, xmm_pattern
        movdqa  xmm6,  xmmword ptr [rax+0]
        movdqa  xmm7,  xmmword ptr [rax+16]
        movdqa  xmm8,  xmmword ptr [rax+32]
        movdqa  xmm9,  xmmword ptr [rax+48]
        movdqa  xmm10, xmmword ptr [rax+64]
        movdqa  xmm11, xmmword ptr [rax+80]
        movdqa  xmm12, xmmword ptr [rax+96]
        movdqa  xmm13, xmmword ptr [rax+112]
        movdqa  xmm14, xmmword ptr [rax+128]
        movdqa  xmm15, xmmword ptr [rax+144]
        stmxcsr dword ptr [rsp+192]

        call    pcuda_sysv_invoke       ; rcx, rdx, r8, r9 passed through untouched
        mov     dword ptr [rsp+196], eax

        xor     r10, r10
        CHECK_GPR rbx, 0B0B0B0B0B0B0B0B0h, 0
        CHECK_GPR rbp, 0B1B1B1B1B1B1B1B1h, 1
        CHECK_GPR rdi, 0B2B2B2B2B2B2B2B2h, 2
        CHECK_GPR rsi, 0B3B3B3B3B3B3B3B3h, 3
        CHECK_GPR r12, 0B4B4B4B4B4B4B4B4h, 4
        CHECK_GPR r13, 0B5B5B5B5B5B5B5B5h, 5
        CHECK_GPR r14, 0B6B6B6B6B6B6B6B6h, 6
        CHECK_GPR r15, 0B7B7B7B7B7B7B7B7h, 7
        lea     rax, xmm_pattern
        CHECK_XMM xmm6, 0
        CHECK_XMM xmm7, 1
        CHECK_XMM xmm8, 2
        CHECK_XMM xmm9, 3
        CHECK_XMM xmm10, 4
        CHECK_XMM xmm11, 5
        CHECK_XMM xmm12, 6
        CHECK_XMM xmm13, 7
        CHECK_XMM xmm14, 8
        CHECK_XMM xmm15, 9
        pushfq
        pop     rcx
        bt      rcx, 10
        jnc     @F
        or      r10, 1 SHL 18
        cld
@@:
        stmxcsr dword ptr [rsp+200]
        mov     ecx, dword ptr [rsp+200]
        cmp     ecx, dword ptr [rsp+192]
        je      @F
        or      r10, 1 SHL 19
@@:
        mov     eax, dword ptr [rsp+196]
        shl     r10, 8
        or      rax, r10

        movaps  xmm6,  xmmword ptr [rsp+32]
        movaps  xmm7,  xmmword ptr [rsp+48]
        movaps  xmm8,  xmmword ptr [rsp+64]
        movaps  xmm9,  xmmword ptr [rsp+80]
        movaps  xmm10, xmmword ptr [rsp+96]
        movaps  xmm11, xmmword ptr [rsp+112]
        movaps  xmm12, xmmword ptr [rsp+128]
        movaps  xmm13, xmmword ptr [rsp+144]
        movaps  xmm14, xmmword ptr [rsp+160]
        movaps  xmm15, xmmword ptr [rsp+176]
        add     rsp, 216
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rsi
        pop     rdi
        pop     rbx
        pop     rbp
        ret
pcuda_test_check_preserved ENDP

; System V: rdi = uint64_t out[16]. Records incoming argument registers and stack
; alignment, then trashes every register it can, sets DF and changes MXCSR.
sysv_echo_clobber PROC
        mov     qword ptr [rdi+0],  rdi
        mov     qword ptr [rdi+8],  rsi
        mov     qword ptr [rdi+16], rdx
        mov     qword ptr [rdi+24], rcx
        mov     qword ptr [rdi+32], r8
        mov     qword ptr [rdi+40], r9
        movq    qword ptr [rdi+48], xmm0
        movq    qword ptr [rdi+56], xmm1
        movq    qword ptr [rdi+64], xmm2
        movq    qword ptr [rdi+72], xmm3
        movq    qword ptr [rdi+80], xmm4
        movq    qword ptr [rdi+88], xmm5
        movq    qword ptr [rdi+96], xmm6
        movq    qword ptr [rdi+104], xmm7
        lea     rax, [rsp+8]            ; System V: rsp + 8 is 16-byte aligned at entry
        and     rax, 15
        mov     qword ptr [rdi+112], rax

        mov     rax, 0DEADBEEFDEADBEEFh
        mov     rbx, rax
        mov     rbp, rax
        mov     rdi, rax
        mov     rsi, rax
        mov     r12, rax
        mov     r13, rax
        mov     r14, rax
        mov     r15, rax
        pcmpeqd xmm6, xmm6
        pcmpeqd xmm7, xmm7
        pcmpeqd xmm8, xmm8
        pcmpeqd xmm9, xmm9
        pcmpeqd xmm10, xmm10
        pcmpeqd xmm11, xmm11
        pcmpeqd xmm12, xmm12
        pcmpeqd xmm13, xmm13
        pcmpeqd xmm14, xmm14
        pcmpeqd xmm15, xmm15
        ldmxcsr dword ptr [clobber_mxcsr]
        std
        ret
sysv_echo_clobber ENDP

; Reads address 0 -> access violation. Clobbers registers first so recovery must restore them.
sysv_fault_read PROC
        mov     rbx, 0DEADBEEFDEADBEEFh
        mov     rsi, rbx
        mov     rdi, rbx
        pcmpeqd xmm6, xmm6
        std
        xor     eax, eax
        mov     rax, qword ptr [rax]
        ret
sysv_fault_read ENDP

; Integer divide by zero.
sysv_div0 PROC
        mov     r12, 0DEADBEEFDEADBEEFh
        xor     ecx, ecx
        mov     eax, 1
        cdq
        idiv    ecx
        ret
sysv_div0 ENDP

; Unbounded recursion touching every page -> stack overflow.
sysv_recurse PROC
        sub     rsp, 4096
        mov     qword ptr [rsp], rax
        call    sysv_recurse
        add     rsp, 4096
        ret
sysv_recurse ENDP

_TEXT ENDS
END
