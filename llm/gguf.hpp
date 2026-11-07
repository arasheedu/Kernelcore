#pragma once

// Minimal GGUF reader (llama.cpp / Ollama model files): metadata, tensor directory, and the
// file's bytes memory-mapped read-only. Only what the real-model check needs.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llm::gguf {

// ggml tensor types used by Llama GGUF files (subset).
enum class Type : uint32_t { F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14 };
const char* typeName(uint32_t t);
// Bytes per block and values per block for the types this project reads.
bool blockInfo(uint32_t t, size_t* block_bytes, size_t* block_values);

struct Tensor {
    std::string          name;
    std::vector<int64_t> ne;      // ne[0] = values per row (contiguous), ne[1] = rows, ...
    uint32_t             type = 0;
    uint64_t             offset = 0;  // from the start of the data section
    const uint8_t*       data = nullptr;
    size_t               bytes = 0;
    int64_t rows() const { return ne.size() > 1 ? ne[1] : 1; }
    int64_t cols() const { return ne.empty() ? 0 : ne[0]; }
};

struct Value {  // scalar metadata as numbers or a string; arrays keep strings or numbers
    std::string              str;
    double                   num = 0;
    std::vector<std::string> strs;
    std::vector<double>      nums;
};

class File {
public:
    explicit File(const std::string& path);  // throws std::runtime_error
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    const std::map<std::string, Value>&  meta() const { return meta_; }
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }
    const Tensor&  tensor(const std::string& name) const;  // throws if missing
    const Tensor*  find(const std::string& name) const;
    double         num(const std::string& key) const;       // throws if missing
    double         num(const std::string& key, double fallback) const;
    std::string    str(const std::string& key) const;

private:
    std::map<std::string, Value>  meta_;
    std::map<std::string, Tensor> tensors_;
    void*                         file_ = nullptr;
    void*                         mapping_ = nullptr;
    const uint8_t*                base_ = nullptr;
    size_t                        size_ = 0;
};

} // namespace llm::gguf
