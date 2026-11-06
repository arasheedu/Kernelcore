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
int dp4a(int a, int b, int c) {  // u8 x s8, as pcuda_dp4a
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
        // (BOOTH_PIN F11); reproduce it so the reference is bit-exact with every pcuda path.
        const float turns = theta * 0.159154943f;
        const float c = std::cos(turns * 6.28318548f);
        const float s = std::sin(turns * 6.28318548f);
        const float x0 = x[2 * idx];
        const float x1 = x[2 * idx + 1];
        x[2 * idx] = x0 * c - x1 * s;
        x[2 * idx + 1] = x0 * s + x1 * c;
    }
}

void attnScores(const float* q, const float* k, float* scores, const AttnDims& d, float scale) {
    for (int h = 0; h < d.n_heads; ++h) {
        const float* qh = q + size_t(h) * d.head_dim;
        for (int t = 0; t < d.ctx; ++t) {
            const float* kt = k + size_t(t) * d.kv_stride + size_t(h / d.group) * d.head_dim;
            float s = 0.0f;
            for (int i = 0; i < d.head_dim; ++i) s += qh[i] * kt[i];
            scores[size_t(h) * d.ctx + t] = s * scale;
        }
    }
}

void softmax(float* x, int n, int rows) {
    for (int row = 0; row < rows; ++row) {
        float* xr = x + size_t(row) * n;
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
            for (int t = 0; t < d.ctx; ++t) s += p[size_t(h) * d.ctx + t] * v[size_t(t) * d.kv_stride + vo];
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
