#include "gguf.hpp"

#include <windows.h>

#include <cstring>
#include <stdexcept>

namespace llm::gguf {

const char* typeName(uint32_t t) {
    switch (t) {
    case 0: return "f32";
    case 1: return "f16";
    case 2: return "q4_0";
    case 8: return "q8_0";
    case 12: return "q4_K";
    case 13: return "q5_K";
    case 14: return "q6_K";
    default: return "?";
    }
}

bool blockInfo(uint32_t t, size_t* bb, size_t* bv) {
    switch (t) {
    case 0: *bb = 4, *bv = 1; return true;
    case 1: *bb = 2, *bv = 1; return true;
    case 2: *bb = 18, *bv = 32; return true;
    case 8: *bb = 34, *bv = 32; return true;
    case 12: *bb = 144, *bv = 256; return true;
    case 13: *bb = 176, *bv = 256; return true;
    case 14: *bb = 210, *bv = 256; return true;
    default: return false;
    }
}

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    void need(size_t n) const {
        if (size_t(end - p) < n) throw std::runtime_error("gguf: truncated file");
    }
    template <class T>
    T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof v);
        p += sizeof v;
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(p), size_t(n));
        p += n;
        return s;
    }
    double number(uint32_t type) {
        switch (type) {
        case 0: return get<uint8_t>();
        case 1: return get<int8_t>();
        case 2: return get<uint16_t>();
        case 3: return get<int16_t>();
        case 4: return get<uint32_t>();
        case 5: return get<int32_t>();
        case 6: return get<float>();
        case 7: return get<uint8_t>();
        case 10: return double(get<uint64_t>());
        case 11: return double(get<int64_t>());
        case 12: return get<double>();
        default: throw std::runtime_error("gguf: unknown value type " + std::to_string(type));
        }
    }
};

} // namespace

File::File(const std::string& path) {
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("gguf: cannot open " + path);
    file_ = f;
    LARGE_INTEGER sz{};
    GetFileSizeEx(f, &sz);
    size_ = size_t(sz.QuadPart);
    mapping_ = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) throw std::runtime_error("gguf: cannot map " + path);
    base_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!base_) throw std::runtime_error("gguf: cannot map view of " + path);

    Reader r{base_, base_ + size_};
    if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic");  // "GGUF"
    const uint32_t version = r.get<uint32_t>();
    if (version < 2) throw std::runtime_error("gguf: unsupported version");
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = r.str();
        const uint32_t type = r.get<uint32_t>();
        Value v;
        if (type == 8) v.str = r.str();
        else if (type == 9) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            for (uint64_t j = 0; j < n; ++j) {
                if (et == 8) v.strs.push_back(r.str());
                else v.nums.push_back(r.number(et));
            }
        } else v.num = r.number(type);
        meta_[key] = std::move(v);
    }
    std::vector<Tensor> list;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.ne.push_back(int64_t(r.get<uint64_t>()));
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        list.push_back(std::move(t));
    }
    const size_t align = size_t(num("general.alignment", 32));
    const size_t header = size_t(r.p - base_);
    const size_t data_start = (header + align - 1) / align * align;
    for (Tensor& t : list) {
        size_t bb = 0, bv = 0;
        if (!blockInfo(t.type, &bb, &bv)) {
            t.bytes = 0;  // unknown type: listed, not readable
        } else {
            int64_t n = 1;
            for (int64_t d : t.ne) n *= d;
            t.bytes = size_t(n) / bv * bb;
        }
        if (data_start + t.offset + t.bytes > size_) throw std::runtime_error("gguf: tensor past end: " + t.name);
        t.data = base_ + data_start + t.offset;
        tensors_[t.name] = std::move(t);
    }
}

File::~File() {
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle(mapping_);
    if (file_) CloseHandle(file_);
}

const Tensor* File::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}
const Tensor& File::tensor(const std::string& name) const {
    if (const Tensor* t = find(name)) return *t;
    throw std::runtime_error("gguf: missing tensor " + name);
}
double File::num(const std::string& key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second.num;
}
double File::num(const std::string& key, double fallback) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? fallback : it->second.num;
}
std::string File::str(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? std::string() : it->second.str;
}

} // namespace llm::gguf
