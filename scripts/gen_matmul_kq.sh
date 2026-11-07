#!/bin/bash
# gen_matmul_kq.sh <q4|q6> <T>: prints matmul_<q>k_q8x<T>.cu, the 4-column kernel with T columns per
# thread (its per-column code repeated T times). matmul_q4k_q8x8.cu / matmul_q6k_q8x8.cu come from
#   bash scripts/gen_matmul_kq.sh q4 8 > kernels/llm/matmul_q4k_q8x8.cu  (the q6 file then gets its
# first 3 comment lines edited). With T = 4 it reproduces the 4-column kernels' code.
q=$1; T=$2; src="$(dirname "$0")/../kernels/llm/matmul_${q}k_q8x4.cu"
rep() { local s="$1"; for ((i=0;i<T;i++)); do echo "${s//@/$i}"; done; }
# header: everything up to the kernel signature, with tile width adjusted
sed -n '1,/^extern "C"/p' $src | sed '$d' | sed "s/for a tile of 4 columns/for a tile of $T columns/; s/once per 4 columns/once per $T columns/; s/once per tile/once per $T-column tile/; s/as in matmul_q4k_q8x4.cu/as in matmul_q4k_q8x4.cu (here $T columns per tile)/"
echo "extern \"C\" __global__ void matmul_${q}k_q8x${T}(const int* qw, const int* xq, float* y, const int* dims, int base)"
# body up to column selection
sed -n '/^extern "C"/,/^        const int c0 = tile \* 4;/p' $src | sed 1d | sed "s/(ncols + 3) \/ 4/(ncols + $((T-1))) \/ $T/; s/tile \* 4;/tile * $T;/"
for ((i=1;i<T;i++)); do echo "        int c$i;"; echo "        if (c0 + $i < ncols) c$i = c0 + $i;"; echo "        else c$i = c$((i-1));"; done
rep '        const int * x@ = xq + c@ * nb * 11;'
rep '        const float * e@ = (const float *)(x@ + nb * 8);'
rep '        float acc@ = 0.0f;'
if [ $q = q4 ]; then
cat <<'E'
        for (int s = 0; s < nsb; ++s) {
            const int wb = (g * nsb + s) * 296;  // 37 words x 8 rows
            const float d = ((const float*)qw)[wb + l];
            const float dmin = ((const float*)qw)[wb + 8 + l];
            const int s0 = qw[wb + 16 + l];
            const int s1 = qw[wb + 24 + l];
            const int s2 = qw[wb + 32 + l];
            const int sclo = s0 & 1061109567;                                      // 0x3F3F3F3F
            const int mlo = s1 & 1061109567;
            const int schi = (s2 & 252645135) | ((s0 >> 2) & 808464432);          // 0x0F0F0F0F, 0x30303030
            const int mhi = ((s2 >> 4) & 252645135) | ((s1 >> 2) & 808464432);
E
rep '            float fd@ = 0.0f;'
rep '            float fm@ = 0.0f;'
cat <<'E'
            for (int j = 0; j < 4; ++j) {
                const int b = s * 8 + j * 2;
E
rep '                int da@ = 0;'
rep '                int db@ = 0;'
cat <<'E'
                for (int w = 0; w < 8; ++w) {
                    const int q = qw[wb + (5 + j * 8 + w) * 8 + l];
                    const int lo = q & 252645135;
                    const int hi = (q >> 4) & 252645135;
E
for ((i=0;i<T;i++)); do echo "                    da$i = pcuda_dp4a(lo, x$i[b * 8 + w], da$i);"; echo "                    db$i = pcuda_dp4a(hi, x$i[b * 8 + 8 + w], db$i);"; done
cat <<'E'
                }
                const int sh = 16 * (j & 1);
                int scw;
                int mw;
                if (j < 2) {
                    scw = sclo;
                    mw = mlo;
                } else {
                    scw = schi;
                    mw = mhi;
                }
                const int sca = (scw >> sh) & 255;
                const int scb = (scw >> (sh + 8)) & 255;
                const int ma = (mw >> sh) & 255;
                const int mb = (mw >> (sh + 8)) & 255;
E
for ((i=0;i<T;i++)); do
echo "                fd$i += (float)(sca * da$i) * e$i[b * 2] + (float)(scb * db$i) * e$i[b * 2 + 2];"
echo "                fm$i += (float)ma * (e$i[b * 2] * e$i[b * 2 + 1]) + (float)mb * (e$i[b * 2 + 2] * e$i[b * 2 + 3]);"; done
echo "            }"
rep '            acc@ += d * fd@ - dmin * fm@;'
echo "        }"
else
cat <<'E'
        for (int s = 0; s < nsb; ++s) {
            const int wb = (g * nsb + s) * 424;  // 53 words x 8 rows
            const float d = ((const float*)qw)[wb + 52 * 8 + l];
            for (int n = 0; n < 2; ++n) {
E
rep '                float p@ = 0.0f;'
cat <<'E'
                for (int t = 0; t < 4; ++t) {
                    const int b = s * 8 + n * 4 + t;
                    const int qlw = 16 * n + 8 * (t & 1);
                    const int lowsh = 4 * (t >> 1);
                    const int hish = 2 * t;
                    const int scw = qw[wb + (48 + 2 * n + (t >> 1)) * 8 + l];
                    const int sh = 16 * (t & 1);
                    const int sca = (scw << (24 - sh)) >> 24;
                    const int scb = (scw << (16 - sh)) >> 24;
E
rep '                    int da@ = 0;'
rep '                    int db@ = 0;'
rep '                    int sa@ = 0;'
cat <<'E'
                    for (int w = 0; w < 4; ++w) {
                        const int lo = (qw[wb + (qlw + w) * 8 + l] >> lowsh) & 252645135;                  // 0x0F0F0F0F
                        const int hi = ((qw[wb + (32 + 8 * n + w) * 8 + l] >> hish) << 4) & 808464432;    // 0x30303030
                        const int q = lo | hi;
E
rep '                        da@ = pcuda_dp4a(q, x@[b * 8 + w], da@);'
rep '                        sa@ = pcuda_dp4a(16843009, x@[b * 8 + w], sa@);'
cat <<'E'
                    }
                    for (int w = 4; w < 8; ++w) {
                        const int lo = (qw[wb + (qlw + w) * 8 + l] >> lowsh) & 252645135;
                        const int hi = ((qw[wb + (32 + 8 * n + w) * 8 + l] >> hish) << 4) & 808464432;
                        const int q = lo | hi;
E
rep '                        db@ = pcuda_dp4a(q, x@[b * 8 + w], db@);'
echo "                    }"
rep '                    const int i@ = sca * (da@ - 32 * sa@) + scb * (db@ - 32 * ((int)e@[b * 2 + 1] - sa@));'
rep '                    p@ += (float)i@ * e@[b * 2];'
echo "                }"
rep '                acc@ += d * p@;'
echo "            }"
echo "        }"
fi
echo "        y[c0 * rows + row] = acc0;"
for ((i=1;i<T;i++)); do echo "        if (c0 + $i < ncols) y[(c0 + $i) * rows + row] = acc$i;"; done
echo "    }"
echo "}"
