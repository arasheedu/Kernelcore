#pragma once

// Loads a Llama GGUF model (Q4_K / Q6_K matrices, f32 norms) into an llm::Model for the decode
// loop, plus the tokenizer's vocabulary for printing tokens.

#include "gguf.hpp"
#include "llm_model.hpp"

#include <string>
#include <vector>

namespace llm {

// The file must outlive the model (token embeddings are read from it on demand).
// Matrices are repacked into the interleaved K-quant streams (ref::repackQ4K / repackQ6K).
Model loadGguf(const gguf::File& f, int max_ctx, Allocator& alloc);

// Token text: GPT-2 byte-level BPE pieces mapped back to bytes (special tokens as written).
std::vector<std::string> ggufVocab(const gguf::File& f);
std::string              tokenText(const std::vector<std::string>& vocab, int id);

} // namespace llm
