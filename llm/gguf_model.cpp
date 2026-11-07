#include "gguf_model.hpp"

#include "llm_reference.hpp"

#include <cstring>
#include <map>
#include <stdexcept>

namespace llm {
namespace {

using gguf::Tensor;

template <class T>
T* upload(Allocator& alloc, const std::vector<T>& v) {
    void* p = alloc.alloc(v.size() * sizeof(T));
    alloc.upload(p, v.data(), v.size() * sizeof(T));
    return static_cast<T*>(p);
}

std::vector<float> floats(const Tensor& t) {
    if (t.type != 0) throw std::runtime_error("expected f32 tensor: " + t.name);
    std::vector<float> v(t.bytes / 4);
    std::memcpy(v.data(), t.data, t.bytes);
    return v;
}

// One K-quant matrix as an interleaved stream in `alloc` memory.
Mat matrix(const Tensor& t, Allocator& alloc, int64_t* file_bytes) {
    const int k = int(t.cols()), rows = int(t.rows());
    if (rows % 8 || k % 256) throw std::runtime_error("matrix shape not supported: " + t.name);
    std::vector<int> qw;
    Weights fmt;
    if (t.type == uint32_t(gguf::Type::Q4_K)) {
        qw.resize(size_t(rows) * (k / 256) * 37);
        ref::repackQ4K(t.data, rows, k, qw.data());
        fmt = Weights::Q4K;
    } else if (t.type == uint32_t(gguf::Type::Q6_K)) {
        qw.resize(size_t(rows) * (k / 256) * 53);
        ref::repackQ6K(t.data, rows, k, qw.data());
        fmt = Weights::Q6K;
    } else {
        throw std::runtime_error(std::string("matrix type not supported: ") + t.name + " " + gguf::typeName(t.type));
    }
    *file_bytes += int64_t(t.bytes);
    return Mat{rows, k, fmt, upload(alloc, qw), nullptr, upload(alloc, std::vector<int>{rows, k})};
}

} // namespace

Model loadGguf(const gguf::File& f, int max_ctx, Allocator& alloc) {
    if (f.str("general.architecture") != "llama") throw std::runtime_error("only llama-architecture GGUF files");
    Config c;
    c.name = f.str("general.name");
    c.hidden = int(f.num("llama.embedding_length"));
    c.inter = int(f.num("llama.feed_forward_length"));
    c.layers = int(f.num("llama.block_count"));
    c.heads = int(f.num("llama.attention.head_count"));
    c.kv_heads = int(f.num("llama.attention.head_count_kv", c.heads));
    c.head_dim = int(f.num("llama.attention.key_length", c.hidden / c.heads));
    c.eps = float(f.num("llama.attention.layer_norm_rms_epsilon", 1e-5));
    c.rope_base = float(f.num("llama.rope.freq_base", 10000.0));
    const Tensor& emb = f.tensor("token_embd.weight");
    c.vocab = int(emb.rows());
    c.max_ctx = max_ctx;
    c.weights = Weights::Gguf;
    if (c.head_dim % 8 || c.hidden % 256 || c.inter % 256) throw std::runtime_error("model shapes not supported");

    Model m;
    m.cfg = c;
    int64_t bytes = 0;
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer L;
        L.qkv_parts = {matrix(f.tensor(p + "attn_q.weight"), alloc, &bytes), matrix(f.tensor(p + "attn_k.weight"), alloc, &bytes),
                       matrix(f.tensor(p + "attn_v.weight"), alloc, &bytes)};
        L.wo = matrix(f.tensor(p + "attn_output.weight"), alloc, &bytes);
        L.gu_parts = {matrix(f.tensor(p + "ffn_gate.weight"), alloc, &bytes), matrix(f.tensor(p + "ffn_up.weight"), alloc, &bytes)};
        L.wd = matrix(f.tensor(p + "ffn_down.weight"), alloc, &bytes);
        L.attn_norm = upload(alloc, floats(f.tensor(p + "attn_norm.weight")));
        L.ffn_norm = upload(alloc, floats(f.tensor(p + "ffn_norm.weight")));
        L.k_cache = upload(alloc, std::vector<float>(size_t(c.kStride()) * c.kvStride(), 0.0f));
        L.v_cache = upload(alloc, std::vector<float>(size_t(c.max_ctx) * c.kvStride(), 0.0f));
        m.layers.push_back(std::move(L));
    }
    m.out_norm = upload(alloc, floats(f.tensor("output_norm.weight")));
    const Tensor* out = f.find("output.weight");
    m.lm_head = matrix(out ? *out : emb, alloc, &bytes);  // tied embeddings when there is no output.weight
    if (const Tensor* rf = f.find("rope_freqs.weight")) m.rope_ff = upload(alloc, floats(*rf));
    const AttnDims d = c.attnDims(c.max_ctx);
    m.attn_dims = upload(alloc, std::vector<int>{d.n_heads, d.ctx, d.head_dim, d.kv_stride, d.group, d.ctx_pad, d.k_stride});
    m.cfg.file_weight_bytes = bytes;

    const int hidden = c.hidden;
    const Tensor* e = &emb;
    m.embed = [e, hidden](int token, float* outv) {
        if (token < 0 || token >= e->rows()) throw std::out_of_range("token id outside the vocabulary");
        const size_t row_bytes = e->bytes / size_t(e->rows());
        const uint8_t* row = e->data + size_t(token) * row_bytes;
        if (e->type == uint32_t(gguf::Type::Q6_K)) ref::dequantRowQ6K(row, hidden, outv);
        else if (e->type == uint32_t(gguf::Type::Q4_K)) ref::dequantRowQ4K(row, hidden, outv);
        else if (e->type == uint32_t(gguf::Type::F32)) std::memcpy(outv, row, size_t(hidden) * 4);
        else throw std::runtime_error("token_embd type not supported");
    };
    return m;
}

std::vector<std::string> ggufVocab(const gguf::File& f) {
    auto it = f.meta().find("tokenizer.ggml.tokens");
    if (it == f.meta().end()) return {};
    return it->second.strs;
}

std::string tokenText(const std::vector<std::string>& vocab, int id) {
    if (id < 0 || size_t(id) >= vocab.size()) return "<?>";
    const std::string& piece = vocab[size_t(id)];
    if (piece.size() > 2 && piece.front() == '<' && piece.back() == '>' && piece[1] == '|') return piece;  // special
    // GPT-2 byte-level BPE: printable bytes map to themselves, the rest to U+0100 + n.
    static const std::map<uint32_t, uint8_t> decoder = [] {
        std::map<uint32_t, uint8_t> m;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            m[keep ? uint32_t(b) : uint32_t(256 + n++)] = uint8_t(b);
        }
        return m;
    }();
    std::string out;
    for (size_t i = 0; i < piece.size();) {  // decode UTF-8 code points
        const uint8_t c0 = uint8_t(piece[i]);
        uint32_t cp;
        size_t len;
        if (c0 < 0x80) cp = c0, len = 1;
        else if ((c0 >> 5) == 6) cp = c0 & 0x1F, len = 2;
        else if ((c0 >> 4) == 14) cp = c0 & 0x0F, len = 3;
        else cp = c0 & 0x07, len = 4;
        for (size_t j = 1; j < len && i + j < piece.size(); ++j) cp = (cp << 6) | (uint8_t(piece[i + j]) & 0x3F);
        i += len;
        auto d = decoder.find(cp);
        out.push_back(d == decoder.end() ? '?' : char(d->second));
    }
    return out;
}

} // namespace llm
