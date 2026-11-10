#include "llm_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llm::ref {

void matvecQ8(const int8_t* q, const float* scales, const float* x, float* y, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            float s = 0.0f;
            const int8_t* qb = q + size_t(row) * k + size_t(b) * 32;
            const float* xb = x + size_t(b) * 32;
            for (int j = 0; j < 32; ++j) s += static_cast<float>(qb[j]) * xb[j];
            acc += s * scales[size_t(row) * nb + b];
        }
        y[row] = acc;
    }
}

void matvecQ4(const uint8_t* qs, const float* scales, const float* x, float* y, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            float s = 0.0f;
            const uint8_t* qb = qs + size_t(row) * (k / 2) + size_t(b) * 16;
            const float* xb = x + size_t(b) * 32;
            for (int j = 0; j < 16; ++j) {
                const int v = qb[j];
                s += static_cast<float>((v & 15) - 8) * xb[j];
                s += static_cast<float>((v >> 4) - 8) * xb[j + 16];
            }
            acc += s * scales[size_t(row) * nb + b];
        }
        y[row] = acc;
    }
}

void matvecQ4x8(const uint8_t* qs, const float* scales, const float* x, float* y, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const size_t blk = size_t(g) * nb + b;
            const uint8_t* qb = qs + blk * 128 + l;
            const float* xb = x + size_t(b) * 32;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;  // same partial sums as the kernel
            for (int j = 0; j < 16; j += 2) {
                const int v0 = qb[j * 8], v1 = qb[j * 8 + 8];
                s0 += static_cast<float>((v0 & 15) - 8) * xb[j];
                s1 += static_cast<float>((v0 >> 4) - 8) * xb[j + 16];
                s2 += static_cast<float>((v1 & 15) - 8) * xb[j + 1];
                s3 += static_cast<float>((v1 >> 4) - 8) * xb[j + 17];
            }
            acc += ((s0 + s2) + (s1 + s3)) * scales[blk * 8 + l];
        }
        y[row] = acc;
    }
}

void repackQ4x8(const uint8_t* qs, const float* scales, uint8_t* qs8, float* scales8, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        for (int b = 0; b < nb; ++b) {
            const size_t blk = size_t(g) * nb + b;
            for (int j = 0; j < 16; ++j) qs8[blk * 128 + size_t(j) * 8 + l] = qs[size_t(row) * (k / 2) + size_t(b) * 16 + j];
            scales8[blk * 8 + l] = scales[size_t(row) * nb + b];
        }
    }
}

void quantizeQ8x4(const float* x, int* xq, float* xd, int count) {
    for (int b = 0; b < count; ++b) {
        const float* xb = x + size_t(b) * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {  // fmaxf as maxss: a > b ? a : b
            const float a = xb[i] > 0.0f - xb[i] ? xb[i] : 0.0f - xb[i];
            amax = amax > a ? amax : a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int w = 0; w < 8; ++w) {
            uint32_t word = 0;
            for (int i = 0; i < 4; ++i) {
                const float t = xb[w * 4 + i] * id;
                int q = t >= 0.0f ? static_cast<int>(t + 0.5f) : static_cast<int>(t - 0.5f);
                q = q > 127 ? 127 : q < -127 ? -127 : q;
                sum += q;
                word |= (static_cast<uint32_t>(q) & 255u) << (i * 8);
            }
            xq[size_t(b) * 8 + w] = static_cast<int>(word);
        }
        xd[size_t(b) * 2] = d;
        xd[size_t(b) * 2 + 1] = static_cast<float>(sum);
    }
}

namespace {
int dp4a(int a, int b, int c) {  // u8 x s8, as kc_dp4a
    for (int i = 0; i < 4; ++i)
        c += ((a >> (8 * i)) & 255) * static_cast<int8_t>(static_cast<uint32_t>(b) >> (8 * i));
    return c;
}
} // namespace

void matvecQ4x8Q8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const int* blk = qw + (size_t(g) * nb + b) * 40;
            int lo = 0, hi = 0;
            for (int w = 0; w < 4; ++w) {
                const int q = blk[w * 8 + l];
                lo = dp4a(q & 0x0F0F0F0F, xq[size_t(b) * 8 + w], lo);
                hi = dp4a((q >> 4) & 0x0F0F0F0F, xq[size_t(b) * 8 + 4 + w], hi);
            }
            float sc;
            std::memcpy(&sc, blk + 32 + l, sizeof sc);
            const int dot = lo + hi - 8 * static_cast<int>(xd[size_t(b) * 2 + 1]);
            acc += static_cast<float>(dot) * (sc * xd[size_t(b) * 2]);
        }
        y[row] = acc;
    }
}

void repackQ4x8Q8(const uint8_t* qs, const float* scales, int* qw, int rows, int k) {
    const int nb = k / 32;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        for (int b = 0; b < nb; ++b) {
            int* blk = qw + (size_t(g) * nb + b) * 40;
            const uint8_t* src = qs + size_t(row) * (k / 2) + size_t(b) * 16;
            for (int w = 0; w < 4; ++w) {
                uint32_t word = 0;
                for (int i = 0; i < 4; ++i) word |= uint32_t(src[w * 4 + i]) << (8 * i);
                blk[w * 8 + l] = static_cast<int>(word);
            }
            std::memcpy(blk + 32 + l, &scales[size_t(row) * nb + b], sizeof(float));
        }
    }
}

void rmsNorm(const float* x, const float* w, float* y, int n, int rows, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float* xr = x + size_t(row) * n;
        float* yr = y + size_t(row) * n;
        float ss = 0.0f;
        for (int j = 0; j < n; ++j) ss += xr[j] * xr[j];
        const float r = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
        for (int j = 0; j < n; ++j) yr[j] = xr[j] * r * w[j];
    }
}

void rope(float* x, int pairs_per_head, int pos, int count, float log2_base) {
    for (int idx = 0; idx < count; ++idx) {
        const int i = idx % pairs_per_head;
        const float theta =
            static_cast<float>(pos) *
            std::exp2(-static_cast<float>(i) / static_cast<float>(pairs_per_head) * log2_base);
        // Booth lowers sinf/cosf(theta) to sinf/cosf((theta * 1/(2pi)) * 2pi) in float
        // (BOOTH_PIN F11); reproduce it so the reference is bit-exact with every Kernelcore path.
        const float turns = theta * 0.159154943f;
        const float c = std::cos(turns * 6.28318548f);
        const float s = std::sin(turns * 6.28318548f);
        const float x0 = x[2 * idx];
        const float x1 = x[2 * idx + 1];
        x[2 * idx] = x0 * c - x1 * s;
        x[2 * idx + 1] = x0 * s + x1 * c;
    }
}

void appendKv(const float* k, const float* v, float* kt, float* vc, int n, int k_stride, int pos) {
    for (int j = 0; j < n; ++j) {
        kt[size_t(j) * k_stride + pos] = k[j];
        vc[size_t(pos) * n + j] = v[j];
    }
}

void attnScores(const float* q, const float* kt, float* scores, const AttnDims& d, float scale) {
    for (int h = 0; h < d.n_heads; ++h) {
        const float* qh = q + size_t(h) * d.head_dim;
        const size_t kb = size_t(h / d.group) * d.head_dim;
        for (int t = 0; t < d.ctx; ++t) {
            float s = 0.0f;
            for (int i = 0; i < d.head_dim; ++i) s += qh[i] * kt[(kb + i) * d.k_stride + t];
            scores[size_t(h) * d.ctx_pad + t] = s * scale;
        }
    }
}

void softmax(float* x, int n, int stride, int rows) {
    for (int row = 0; row < rows; ++row) {
        float* xr = x + size_t(row) * stride;
        float m = xr[0];
        for (int j = 1; j < n; ++j) m = std::max(m, xr[j]);
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = std::exp2((xr[j] - m) * 1.44269504f);
            xr[j] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < n; ++j) xr[j] = xr[j] * inv;
    }
}

void attnValues(const float* p, const float* v, float* out, const AttnDims& d) {
    for (int h = 0; h < d.n_heads; ++h) {
        for (int i = 0; i < d.head_dim; ++i) {
            const size_t vo = size_t(h / d.group) * d.head_dim + i;
            float s = 0.0f;
            for (int t = 0; t < d.ctx; ++t) s += p[size_t(h) * d.ctx_pad + t] * v[size_t(t) * d.kv_stride + vo];
            out[size_t(h) * d.head_dim + i] = s;
        }
    }
}

void siluMul(const float* g, const float* u, float* out, int count) {
    for (int i = 0; i < count; ++i) out[i] = g[i] / (1.0f + std::exp2(-g[i] * 1.44269504f)) * u[i];
}

void add(float* x, const float* t, int n) {
    for (int i = 0; i < n; ++i) x[i] = x[i] + t[i];
}

} // namespace llm::ref

namespace llm::ref {

void addRmsNorm(float* x, const float* t, const float* w, float* y, int n, float eps) {
    add(x, t, n);
    rmsNorm(x, w, y, n, 1, eps);
}

void siluMulQ8x4(const float* g, const float* u, int* xq, float* xd, int count) {
    std::vector<float> h(size_t(count) * 32);
    siluMul(g, u, h.data(), count * 32);
    quantizeQ8x4(h.data(), xq, xd, count);
}

} // namespace llm::ref

namespace llm::ref {

void addSumsq(float* x, const float* t, int* q, int nb) {
    float* ss = reinterpret_cast<float*>(q + nb * 10);
    for (int b = 0; b < nb; ++b) {
        float s = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = x[b * 32 + i] + t[b * 32 + i];
            x[b * 32 + i] = v;
            s += v * v;
        }
        ss[b] = s;
    }
}

void normQ8x4(const float* x, const float* w, int* q, int nb, float eps) {
    const float* ss = reinterpret_cast<const float*>(q + nb * 10);
    float tot = 0.0f;
    for (int j = 0; j < nb; ++j) tot += ss[j];
    const float r = 1.0f / std::sqrt(tot / static_cast<float>(nb * 32) + eps);
    std::vector<float> h(size_t(nb) * 32);
    for (size_t i = 0; i < h.size(); ++i) h[i] = x[i] * r * w[i];
    quantizeQ8x4(h.data(), q, reinterpret_cast<float*>(q + nb * 8), nb);
}

} // namespace llm::ref

// ---- GGUF K-quants (Q4_K, Q6_K) ------------------------------------------------

namespace llm::ref {

float f16ToF32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: renormalize
            int e = -1;
            uint32_t m = man;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            bits = sign | uint32_t(127 - 15 - e) << 23 | (m & 0x3FFu) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | man << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | man << 13;
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

namespace {
uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }
int rd32(const uint8_t* p) {
    int v;
    std::memcpy(&v, p, 4);
    return v;
}
int f32bits(float f) {
    int v;
    std::memcpy(&v, &f, 4);
    return v;
}
float bitsF32(int v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
// ggml get_scale_min_k4
void scaleMinK4(int j, const uint8_t* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
int sar(int v, int s) { return v >> s; }                                                   // arithmetic
int shl(int v, int s) { return static_cast<int>(static_cast<uint32_t>(v) << s); }        // no UB
} // namespace

void dequantRowQ4K(const uint8_t* src, int k, float* out) {
    for (int s = 0; s < k / 256; ++s) {
        const uint8_t* b = src + size_t(s) * 144;
        const float d = f16ToF32(rd16(b)), dmin = f16ToF32(rd16(b + 2));
        const uint8_t* scales = b + 4;
        const uint8_t* q = b + 16;
        float* y = out + size_t(s) * 256;
        for (int j = 0; j < 4; ++j) {
            int sc, m;
            scaleMinK4(2 * j, scales, &sc, &m);
            const float d1 = d * float(sc), m1 = dmin * float(m);
            scaleMinK4(2 * j + 1, scales, &sc, &m);
            const float d2 = d * float(sc), m2 = dmin * float(m);
            for (int l = 0; l < 32; ++l) y[64 * j + l] = d1 * float(q[32 * j + l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) y[64 * j + 32 + l] = d2 * float(q[32 * j + l] >> 4) - m2;
        }
    }
}

void dequantRowQ6K(const uint8_t* src, int k, float* out) {
    for (int s = 0; s < k / 256; ++s) {
        const uint8_t* b = src + size_t(s) * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = f16ToF32(rd16(b + 208));
        float* y = out + size_t(s) * 256;
        for (int n = 0; n < 2; ++n) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[64 * n + l] & 0xF) | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32;
                y[128 * n + l] = d * float(sc[8 * n + is + 0]) * float(q1);
                y[128 * n + l + 32] = d * float(sc[8 * n + is + 2]) * float(q2);
                y[128 * n + l + 64] = d * float(sc[8 * n + is + 4]) * float(q3);
                y[128 * n + l + 96] = d * float(sc[8 * n + is + 6]) * float(q4);
            }
        }
    }
}

void repackQ4K(const uint8_t* src, int rows, int k, int* qw) {
    const int nsb = k / 256;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* b = src + (size_t(row) * nsb + s) * 144;
            int* dst = qw + (size_t(g) * nsb + s) * 296;
            dst[0 * 8 + l] = f32bits(f16ToF32(rd16(b)));
            dst[1 * 8 + l] = f32bits(f16ToF32(rd16(b + 2)));
            for (int w = 0; w < 3; ++w) dst[(2 + w) * 8 + l] = rd32(b + 4 + 4 * w);
            for (int w = 0; w < 32; ++w) dst[(5 + w) * 8 + l] = rd32(b + 16 + 4 * w);
        }
    }
}

void repackQ6K(const uint8_t* src, int rows, int k, int* qw) {
    const int nsb = k / 256;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        for (int s = 0; s < nsb; ++s) {
            const uint8_t* b = src + (size_t(row) * nsb + s) * 210;
            int* dst = qw + (size_t(g) * nsb + s) * 424;
            for (int w = 0; w < 52; ++w) dst[w * 8 + l] = rd32(b + 4 * w);  // ql, qh, scales
            dst[52 * 8 + l] = f32bits(f16ToF32(rd16(b + 208)));
        }
    }
}

void matvecQ4KQ8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k) {
    const int nsb = k / 256;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int* blk = qw + (size_t(g) * nsb + s) * 296;
            const float d = bitsF32(blk[l]), dmin = bitsF32(blk[8 + l]);
            const int s0 = blk[16 + l], s1 = blk[24 + l], s2 = blk[32 + l];
            const int sclo = s0 & 0x3F3F3F3F, mlo = s1 & 0x3F3F3F3F;
            const int schi = (s2 & 0x0F0F0F0F) | (sar(s0, 2) & 0x30303030);
            const int mhi = (sar(s2, 4) & 0x0F0F0F0F) | (sar(s1, 2) & 0x30303030);
            float fd = 0.0f, fm = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const int b = s * 8 + j * 2;
                int da = 0, db = 0;
                for (int w = 0; w < 8; ++w) {
                    const int q = blk[(5 + j * 8 + w) * 8 + l];
                    da = dp4a(q & 0x0F0F0F0F, xq[b * 8 + w], da);
                    db = dp4a(sar(q, 4) & 0x0F0F0F0F, xq[b * 8 + 8 + w], db);
                }
                const int sh = 16 * (j & 1);
                const int scw = j < 2 ? sclo : schi, mw = j < 2 ? mlo : mhi;
                const int sca = sar(scw, sh) & 255, scb = sar(scw, sh + 8) & 255;
                const int ma = sar(mw, sh) & 255, mb = sar(mw, sh + 8) & 255;
                fd += float(sca * da) * xd[b * 2] + float(scb * db) * xd[b * 2 + 2];
                fm += float(ma) * (xd[b * 2] * xd[b * 2 + 1]) + float(mb) * (xd[b * 2 + 2] * xd[b * 2 + 3]);
            }
            acc += d * fd - dmin * fm;
        }
        y[row] = acc;
    }
}

void matvecQ6KQ8(const int* qw, const int* xq, const float* xd, float* y, int rows, int k) {
    const int nsb = k / 256;
    auto byteS = [](int w, int i) { return sar(shl(w, 24 - 8 * i), 24); };  // signed byte i
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int* blk = qw + (size_t(g) * nsb + s) * 424;
            const float d = bitsF32(blk[52 * 8 + l]);
            for (int n = 0; n < 2; ++n) {
                const int b = s * 8 + n * 4;
                int da[4] = {}, db[4] = {}, sa[4] = {};
                for (int w = 0; w < 8; ++w) {
                    const int qa = blk[(16 * n + w) * 8 + l], qc = blk[(16 * n + 8 + w) * 8 + l];
                    const int h = blk[(32 + 8 * n + w) * 8 + l];
                    const int q[4] = {(qa & 0x0F0F0F0F) | (shl(h, 4) & 0x30303030),
                                      (qc & 0x0F0F0F0F) | (shl(h, 2) & 0x30303030),
                                      (sar(qa, 4) & 0x0F0F0F0F) | (h & 0x30303030),
                                      (sar(qc, 4) & 0x0F0F0F0F) | (sar(h, 2) & 0x30303030)};
                    for (int t = 0; t < 4; ++t) {
                        const int x = xq[b * 8 + 8 * t + w];
                        if (w < 4) {
                            da[t] = dp4a(q[t], x, da[t]);
                            sa[t] = dp4a(0x01010101, x, sa[t]);
                        } else {
                            db[t] = dp4a(q[t], x, db[t]);
                        }
                    }
                }
                const int sw0 = blk[(48 + 2 * n) * 8 + l], sw1 = blk[(49 + 2 * n) * 8 + l];
                int is[4];
                for (int t = 0; t < 4; ++t) {
                    const int sw = t < 2 ? sw0 : sw1, lo = 2 * (t & 1);
                    const int sb = int(xd[(b + t) * 2 + 1]) - sa[t];
                    is[t] = byteS(sw, lo) * (da[t] - 32 * sa[t]) + byteS(sw, lo + 1) * (db[t] - 32 * sb);
                }
                acc += d * (float(is[0]) * xd[b * 2] + float(is[1]) * xd[b * 2 + 2] + float(is[2]) * xd[b * 2 + 4] +
                            float(is[3]) * xd[b * 2 + 6]);
            }
        }
        y[row] = acc;
    }
}

void quantizeQ8K(const float* x, int* q, int nsb) {
    float* sc = reinterpret_cast<float*>(q + size_t(nsb) * 64);
    int*   bs = q + size_t(nsb) * 65;
    for (int s = 0; s < nsb; ++s) {
        const float* xs = x + size_t(s) * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) {  // fmaxf as maxss: a > b ? a : b
            const float a = xs[i] > 0.0f - xs[i] ? xs[i] : 0.0f - xs[i];
            amax = amax > a ? amax : a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (int g = 0; g < 16; ++g) {
            int sum = 0;
            for (int w = 0; w < 4; ++w) {
                uint32_t word = 0;
                for (int i = 0; i < 4; ++i) {
                    const float t = xs[g * 16 + w * 4 + i] * id;
                    int v = t >= 0.0f ? static_cast<int>(t + 0.5f) : static_cast<int>(t - 0.5f);
                    v = v > 127 ? 127 : v < -127 ? -127 : v;
                    sum += v;
                    word |= (static_cast<uint32_t>(v) & 255u) << (i * 8);
                }
                q[size_t(s) * 64 + g * 4 + w] = static_cast<int>(word);
            }
            bs[size_t(s) * 16 + g] = sum;
        }
        sc[s] = d;
        uint32_t pw[4] = {};  // sums of 32 as bytes: B >> 6 (sums 0-3, 4-7), then B & 63
        for (int i = 0; i < 8; ++i) {
            const int b = bs[size_t(s) * 16 + 2 * i] + bs[size_t(s) * 16 + 2 * i + 1];
            pw[i / 4] |= (static_cast<uint32_t>(sar(b, 6)) & 255u) << (8 * (i % 4));
            pw[2 + i / 4] |= static_cast<uint32_t>(b & 63) << (8 * (i % 4));
        }
        for (int i = 0; i < 4; ++i) q[size_t(nsb) * 81 + size_t(s) * 4 + i] = static_cast<int>(pw[i]);
    }
}

void matvecQ4KQ8K(const int* qw, const int* xq, float* y, int rows, int k) {
    const int    nsb = k / 256;
    const float* xd = reinterpret_cast<const float*>(xq + size_t(nsb) * 64);
    const int*   bs = xq + size_t(nsb) * 65;
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int* blk = qw + (size_t(g) * nsb + s) * 296;
            const float d = bitsF32(blk[l]), dmin = bitsF32(blk[8 + l]);
            const int s0 = blk[16 + l], s1 = blk[24 + l], s2 = blk[32 + l];
            const int sclo = s0 & 0x3F3F3F3F, mlo = s1 & 0x3F3F3F3F;
            const int schi = (s2 & 0x0F0F0F0F) | (sar(s0, 2) & 0x30303030);
            const int mhi = (sar(s2, 4) & 0x0F0F0F0F) | (sar(s1, 2) & 0x30303030);
            int is = 0, im = 0;
            for (int j = 0; j < 4; ++j) {
                const int* xs = xq + size_t(s) * 64 + j * 16;  // sub-blocks 2j and 2j+1
                int da = 0, db = 0;
                for (int w = 0; w < 8; ++w) {
                    const int q = blk[(5 + j * 8 + w) * 8 + l];
                    da = dp4a(q & 0x0F0F0F0F, xs[w], da);
                    db = dp4a(sar(q, 4) & 0x0F0F0F0F, xs[8 + w], db);
                }
                const int sh = 16 * (j & 1);
                const int scw = j < 2 ? sclo : schi, mw = j < 2 ? mlo : mhi;
                is += (sar(scw, sh) & 255) * da + (sar(scw, sh + 8) & 255) * db;
                const int* b = bs + size_t(s) * 16 + j * 4;
                im += (sar(mw, sh) & 255) * (b[0] + b[1]) + (sar(mw, sh + 8) & 255) * (b[2] + b[3]);
            }
            acc += (d * xd[s]) * float(is) - (dmin * xd[s]) * float(im);
        }
        y[row] = acc;
    }
}

void matvecQ6KQ8K(const int* qw, const int* xq, float* y, int rows, int k) {
    const int    nsb = k / 256;
    const float* xd = reinterpret_cast<const float*>(xq + size_t(nsb) * 64);
    const int*   bs = xq + size_t(nsb) * 65;
    auto byteS = [](int w, int i) { return sar(shl(w, 24 - 8 * i), 24); };  // signed byte i
    for (int row = 0; row < rows; ++row) {
        const int g = row / 8, l = row % 8;
        float acc = 0.0f;
        for (int s = 0; s < nsb; ++s) {
            const int* blk = qw + (size_t(g) * nsb + s) * 424;
            const float d = bitsF32(blk[52 * 8 + l]);
            int is = 0;
            for (int n = 0; n < 2; ++n) {
                for (int t = 0; t < 4; ++t) {
                    const int* xs = xq + size_t(s) * 64 + (n * 4 + t) * 8;  // this block of 32
                    int da = 0, db = 0;
                    for (int w = 0; w < 8; ++w) {
                        const int qa = blk[(16 * n + 8 * (t & 1) + w) * 8 + l];
                        const int h = blk[(32 + 8 * n + w) * 8 + l];
                        const int q = (sar(qa, 4 * (t >> 1)) & 0x0F0F0F0F) | (shl(sar(h, 2 * t), 4) & 0x30303030);
                        if (w < 4) da = dp4a(q, xs[w], da);
                        else db = dp4a(q, xs[w], db);
                    }
                    const int sw = blk[(48 + 2 * n + (t >> 1)) * 8 + l], lo = 2 * (t & 1);
                    const int* b = bs + size_t(s) * 16 + (n * 4 + t) * 2;
                    is += byteS(sw, lo) * (da - 32 * b[0]) + byteS(sw, lo + 1) * (db - 32 * b[1]);
                }
            }
            acc += (d * xd[s]) * float(is);
        }
        y[row] = acc;
    }
}

void ropeFF(float* x, const float* ff, int pairs_per_head, int pos, int count, float log2_base) {
    for (int idx = 0; idx < count; ++idx) {
        const int i = idx % pairs_per_head;
        const float theta = static_cast<float>(pos) *
                            std::exp2(-static_cast<float>(i) / static_cast<float>(pairs_per_head) * log2_base) / ff[i];
        const float turns = theta * 0.159154943f;  // Booth lowers sin/cos this way (BOOTH_PIN F11)
        const float c = std::cos(turns * 6.28318548f);
        const float s = std::sin(turns * 6.28318548f);
        const float x0 = x[2 * idx], x1 = x[2 * idx + 1];
        x[2 * idx] = x0 * c - x1 * s;
        x[2 * idx + 1] = x0 * s + x1 * c;
    }
}

} // namespace llm::ref

namespace llm::ref {

void mulRows(const float* x, const float* w, float* y, int n, int count) {
    for (int i = 0; i < count; ++i) y[i] = x[i] * w[i % n];
}

void ropeGgml(const float* x, float* y, const int* pos, const float* ff, int head_dim, int n_head, int n_tokens,
              float log2_base, bool neox) {
    const int pairs = head_dim / 2, count = pairs * n_head * n_tokens;
    for (int idx = 0; idx < count; ++idx) {
        const int t = idx / (pairs * n_head), i = idx % pairs;
        const float theta = static_cast<float>(pos[t]) *
                            std::exp2(-static_cast<float>(i) / static_cast<float>(pairs) * log2_base) / ff[i];
        const float turns = theta * 0.159154943f;  // Booth's sin/cos lowering (BOOTH_PIN F11)
        const float c = std::cos(turns * 6.28318548f), s = std::sin(turns * 6.28318548f);
        const int head = idx / pairs;  // (token, head) row
        const int a = neox ? head * head_dim + i : 2 * idx, b = neox ? a + pairs : a + 1;
        const float x0 = x[a], x1 = x[b];
        y[a] = x0 * c - x1 * s;
        y[b] = x0 * s + x1 * c;
    }
}

} // namespace llm::ref

namespace llm::ref {

namespace {
void quantizeBlockInto(const float* y, int* qr, int nb, int bl) {  // as quantize_q8x4, one block
    std::vector<int> words(8);
    std::vector<float> xd(2);
    quantizeQ8x4(y, words.data(), xd.data(), 1);
    for (int k = 0; k < 8; ++k) qr[bl * 8 + k] = words[size_t(k)];
    float* d = reinterpret_cast<float*>(qr + nb * 8);
    d[bl * 2] = xd[0];
    d[bl * 2 + 1] = xd[1];
}
} // namespace

void addSumsqF(const float* a, const float* b, float* out, int* q, int n, int rows) {
    const int nb = n / 32;
    for (int row = 0; row < rows; ++row) {
        float* ss = reinterpret_cast<float*>(q + size_t(row) * nb * 11 + nb * 10);
        for (int bl = 0; bl < nb; ++bl) {
            float s = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const size_t o = size_t(row) * n + size_t(bl) * 32 + i;
                const float v = a[o] + b[o];
                out[o] = v;
                s += v * v;
            }
            ss[bl] = s;
        }
    }
}

void sumsqPart(const float* x, int* q, int n, int rows) {
    const int nb = n / 32;
    for (int row = 0; row < rows; ++row) {
        float* ss = reinterpret_cast<float*>(q + size_t(row) * nb * 11 + nb * 10);
        for (int bl = 0; bl < nb; ++bl) {
            float s = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float v = x[size_t(row) * n + size_t(bl) * 32 + i];
                s += v * v;
            }
            ss[bl] = s;
        }
    }
}

void normMulQ8(const float* x, const float* w, float* y, int* q, int n, int rows, float eps) {
    const int nb = n / 32;
    for (int row = 0; row < rows; ++row) {
        int* qr = q + size_t(row) * nb * 11;
        const float* ss = reinterpret_cast<const float*>(qr + nb * 10);
        float tot = 0.0f;
        for (int j = 0; j < nb; ++j) tot += ss[j];
        const float r = 1.0f / std::sqrt(tot / static_cast<float>(n) + eps);
        for (int bl = 0; bl < nb; ++bl) {
            float* yb = y + size_t(row) * n + size_t(bl) * 32;
            const float* xb = x + size_t(row) * n + size_t(bl) * 32;
            for (int i = 0; i < 32; ++i) yb[i] = xb[i] * r * w[bl * 32 + i];
            quantizeBlockInto(yb, qr, nb, bl);
        }
    }
}

void normMulQ8K(const float* x, const float* w, float* y, int* q, const int* sq, int n, int rows, float eps) {
    const int nb = n / 32;
    for (int row = 0; row < rows; ++row) {
        const float* ss = reinterpret_cast<const float*>(sq + size_t(row) * nb * 11 + nb * 10);
        float tot = 0.0f;
        for (int j = 0; j < nb; ++j) tot += ss[j];
        const float r = 1.0f / std::sqrt(tot / static_cast<float>(n) + eps);
        for (int i = 0; i < n; ++i) y[size_t(row) * n + i] = x[size_t(row) * n + i] * r * w[i];
        quantizeQ8K(y + size_t(row) * n, q + size_t(row) * (n / 256 * 85), n / 256);
    }
}
void siluMulQ8K(const float* g, const float* u, float* y, int* q, int n, int cols) {
    for (size_t i = 0; i < size_t(n) * cols; ++i) y[i] = g[i] / (1.0f + std::exp2(-g[i] * 1.44269504f)) * u[i];
    for (int c = 0; c < cols; ++c) quantizeQ8K(y + size_t(c) * n, q + size_t(c) * (n / 256 * 85), n / 256);
}
void siluMulQ8F(const float* g, const float* u, float* y, int* q, int n, int cols) {
    const int nb = n / 32;
    for (int c = 0; c < cols; ++c) {
        int* qc = q + size_t(c) * nb * 11;
        for (int bl = 0; bl < nb; ++bl) {
            const size_t o = size_t(c) * n + size_t(bl) * 32;
            for (int i = 0; i < 32; ++i) y[o + i] = g[o + i] / (1.0f + std::exp2(-g[o + i] * 1.44269504f)) * u[o + i];
            quantizeBlockInto(y + o, qc, nb, bl);
        }
    }
}


uint16_t f32ToF16(float x) {
    const uint32_t sign = x < 0.0f ? 0x8000u : 0u;  // as the kernel: -0 gives +0
    const float a = std::fabs(x);
    if (!(a < 65520.0f)) return uint16_t(sign | 0x7bffu);
    if (a < 6.103515625e-05f) return uint16_t(sign | uint32_t(std::nearbyint(a * 16777216.0f)));  // subnormal
    int e = 0;
    const float f = std::frexp(a, &e);  // a = f * 2^e, f in [0.5, 1)
    const uint32_t m = uint32_t(std::nearbyint((f * 2.0f - 1.0f) * 1024.0f));
    return uint16_t(sign | ((uint32_t(e - 1 + 15) << 10) + m));
}
void repackF16(const uint16_t* w, int rows, int k, int* out) {
    const int kw = k / 2;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < kw; ++j)
            out[(size_t(r / 8) * kw + j) * 8 + r % 8] =
                int(uint32_t(w[size_t(r) * k + 2 * j]) | (uint32_t(w[size_t(r) * k + 2 * j + 1]) << 16));
}
void unpackF16(const int* stream, int rows, int k, uint16_t* w) {
    const int kw = k / 2;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < kw; ++j) {
            const uint32_t v = uint32_t(stream[(size_t(r / 8) * kw + j) * 8 + r % 8]);
            w[size_t(r) * k + 2 * j] = uint16_t(v & 0xFFFF);
            w[size_t(r) * k + 2 * j + 1] = uint16_t(v >> 16);
        }
}
void matmulF16(const uint16_t* w, const float* x, float* y, int rows, int k, int ncols) {
    for (int c = 0; c < ncols; ++c)
        for (int r = 0; r < rows; ++r) {
            float a = 0.0f;
            for (int j = 0; j < k; j += 2)
                a += f16ToF32(w[size_t(r) * k + j]) * x[size_t(c) * k + j] + f16ToF32(w[size_t(r) * k + j + 1]) * x[size_t(c) * k + j + 1];
            y[size_t(c) * rows + r] = a;
        }
}
void setRowsF16(const float* src, const int64_t* idx, uint16_t* dst, int pairs, int rows, int src_row, int dst_row,
                int dst_rows) {
    for (int r = 0; r < rows; ++r) {
        const int row = int(idx[r]);  // the low 32-bit word, as the kernel reads it
        if (row < 0 || row >= dst_rows) continue;
        for (int j = 0; j < 2 * pairs; ++j) dst[size_t(row) * dst_row * 2 + j] = f32ToF16(src[size_t(r) * src_row + j]);
    }
}
void flashAttnF16(const float* q, const uint16_t* k, const uint16_t* v, const uint16_t* mask, float* out,
                  const FaDims& d, float scale) {
    std::vector<float> s(size_t(d.T));
    for (int i = 0; i < d.N; ++i) {
        for (int h = 0; h < d.H; ++h) {
            const int kvh = h / d.group;
            const float* qh = q + size_t(i) * d.q_tok + size_t(h) * d.q_head;
            float mx = -INFINITY;
            for (int t = 0; t < d.T; ++t) {
                const uint16_t* kr = k + (size_t(t) * d.k_row + size_t(kvh) * d.k_head) * 2;
                float a = 0.0f;
                for (int j = 0; j < d.D; ++j) a += qh[j] * f16ToF32(kr[j]);
                const float mv = d.has_mask ? f16ToF32(mask[size_t(i) * d.m_row * 2 + t]) : 0.0f;
                s[t] = a * scale + mv;
                mx = std::max(mx, s[t]);
            }
            float sum = 0.0f;
            for (int t = 0; t < d.T; ++t) sum += (s[t] = std::exp2((s[t] - mx) * 1.44269504f));
            for (int dd = 0; dd < d.D; ++dd) {
                float a = 0.0f;
                for (int t = 0; t < d.T; ++t) a += s[t] * f16ToF32(v[(size_t(t) * d.v_row + size_t(kvh) * d.v_head) * 2 + dd]);
                out[(size_t(i) * d.H + h) * d.D + dd] = a / sum;
            }
        }
    }
}

} // namespace llm::ref
