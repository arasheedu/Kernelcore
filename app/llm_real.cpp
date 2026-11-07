// pcuda_llm_real: runs a real Llama GGUF model (e.g. Ollama's llama3.2, Q4_K_M) through pcuda's
// CUDA C kernels on the CPU, and compares it with llama.cpp on the same file.
//
//   pcuda_llm_real --gguf FILE --list
//   pcuda_llm_real --gguf FILE --ref REF [--workers N] [--graphs on|off] [--mirror K]
//
// REF (written by scripts/real_model_check.ps1 from llama-server):
//   prompt <id> <id> ...
//   gen <id> <logprob> <top id> <top logprob> ...      one line per generated token (greedy)
// Checks:
//   1. mirror: the first K prompt positions on pcuda and on the C++ mirror of the kernels give
//      bit-identical logits;
//   2. teacher forcing: fed llama.cpp's tokens, our top token and log-probabilities at each
//      generated position against llama.cpp's;
//   3. free run: greedy generation from the prompt, compared with llama.cpp's text, and timed.
#include "gguf_model.hpp"
#include "llm_kernels.hpp"
#include "llm_model.hpp"
#include "llm_reference.hpp"

#include <pcuda/api.hpp>

#include <algorithm>
#include <bit>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct RefToken {
    int                                id = 0;
    double                             logprob = 0;
    std::vector<std::pair<int, double>> top;
};

struct Ref {
    std::vector<int>      prompt;
    std::vector<RefToken> gen;
};

Ref readRef(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    Ref r;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string kind;
        ss >> kind;
        if (kind == "prompt") {
            int id;
            while (ss >> id) r.prompt.push_back(id);
        } else if (kind == "gen") {
            RefToken t;
            ss >> t.id >> t.logprob;
            int id;
            double lp;
            while (ss >> id >> lp) t.top.push_back({id, lp});
            r.gen.push_back(t);
        }
    }
    if (r.prompt.empty() || r.gen.empty()) throw std::runtime_error("reference file has no prompt or no tokens");
    return r;
}

std::vector<double> logSoftmax(const std::vector<float>& z) {
    double mx = -1e300;
    for (float v : z) mx = std::max(mx, double(v));
    double s = 0;
    for (float v : z) s += std::exp(double(v) - mx);
    const double lse = mx + std::log(s);
    std::vector<double> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = double(z[i]) - lse;
    return out;
}

int argmax(const std::vector<float>& z) { return int(std::max_element(z.begin(), z.end()) - z.begin()); }

std::string text(const std::vector<std::string>& vocab, const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) s += llm::tokenText(vocab, id);
    for (char& c : s)
        if (c == '\n') c = ' ';  // one line
    return s;
}

// --bench-matvec: GB/s of each 8-bit-activation matvec kernel on an FFN-sized matrix (rows x k),
// chunked over the pool like the decode loop. Weights are random bytes with valid scales.
int benchMatvec(unsigned workers) {
    const int rows = 8192, k = 3072, nb = k / 32;
    std::mt19937 rng(7);
    auto randomInts = [&](size_t n) {
        std::vector<int> v(n);
        for (auto& x : v) x = int(rng());
        return v;
    };
    struct K {
        const char* name;
        int         words;  // per row per 256 weights
        size_t      file_bytes;
    };
    const K kernels[] = {{"matvec_q4x8_q8", 40, size_t(rows) * k * 5 / 8},
                         {"matvec_q4k_q8", 37, size_t(rows) * k / 256 * 144},
                         {"matvec_q6k_q8", 53, size_t(rows) * k / 256 * 210}};
    llm::PcudaAllocator pa;
    std::vector<int> xq(size_t(nb) * 8);
    std::vector<float> xd(size_t(nb) * 2);
    {
        std::vector<float> x(static_cast<size_t>(k));
        for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 500.0f;
        llm::ref::quantizeQ8x4(x.data(), xq.data(), xd.data(), nb);
    }
    auto* dxq = static_cast<int*>(pa.alloc(xq.size() * 4));
    auto* dxd = static_cast<float*>(pa.alloc(xd.size() * 4));
    auto* dy = static_cast<float*>(pa.alloc(size_t(rows) * 4));
    auto* dd = static_cast<int*>(pa.alloc(8));
    const int dims[2] = {rows, k};
    pa.upload(dxq, xq.data(), xq.size() * 4);
    pa.upload(dxd, xd.data(), xd.size() * 4);
    pa.upload(dd, dims, sizeof dims);
    std::printf("matvec %d x %d, chunked over %u workers (median of 5 runs of 50 launches)\n", rows, k, workers);
    for (const K& kn : kernels) {
        const size_t ints = size_t(rows) * k / 256 * kn.words;
        std::vector<int> qw = randomInts(ints);
        if (kn.words == 40) {  // q4x8q8: finite scales in words 32..39 of each 40
            for (size_t i = 0; i < ints; ++i)
                if (i % 40 >= 32) qw[i] = std::bit_cast<int>(0.004f);
        } else if (kn.words == 37) {  // q4k: d, dmin are words 0, 1 of each 37 (x8 interleaved)
            for (size_t i = 0; i < ints; ++i)
                if (i % 296 < 16) qw[i] = std::bit_cast<int>(0.004f);
        } else {  // q6k: d is word 52 of 53
            for (size_t i = 0; i < ints; ++i)
                if (i % 424 >= 416) qw[i] = std::bit_cast<int>(0.004f);
        }
        auto* dqw = static_cast<int*>(pa.alloc(ints * 4));
        pa.upload(dqw, qw.data(), ints * 4);
        const llm::MatvecQ4Q8Args a{dqw, dxq, dxd, dy, dd, rows, k};
        std::vector<double> runs;
        for (int r = 0; r < 6; ++r) {
            const auto t = Clock::now();
            for (int i = 0; i < 50; ++i)
                if (pcudaLaunchKernelChunked(kn.name, &a, workers, 16) != pcuda::Error::Success)
                    throw std::runtime_error(std::string(kn.name) + ": " + pcudaGetLastErrorDetail());
            if (r) runs.push_back(std::chrono::duration<double>(Clock::now() - t).count() / 50);
        }
        std::sort(runs.begin(), runs.end());
        const double s = runs[runs.size() / 2];
        std::printf("  %-16s %8.1f us  stream %6.1f GB/s  (file format %6.1f GB/s)\n", kn.name, s * 1e6,
                    double(ints) * 4 / s / 1e9, double(kn.file_bytes) / s / 1e9);
    }
    return 0;
}

int run(int argc, char** argv) {
    std::string path, ref_path;
    bool list = false, graphs = true;
    unsigned workers = 0;
    int mirror = 2;
    bool bench_mv = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--gguf") path = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "--list") list = true;
        else if (a == "--bench-matvec") bench_mv = true;
        else if (a == "--workers") workers = unsigned(std::atoi(next().c_str()));
        else if (a == "--mirror") mirror = std::atoi(next().c_str());
        else if (a == "--graphs") graphs = next() == "on";
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (bench_mv) {
        if (pcudaInit() != pcuda::Error::Success || llm::registerKernels() != pcuda::Error::Success)
            throw std::runtime_error("pcuda init");
        pcuda::DeviceProperties pr{};
        pcudaGetDeviceProperties(&pr, 0);
        const int r = benchMatvec(workers ? workers : pr.worker_count + 1);
        pcudaShutdown();
        return r;
    }
    if (path.empty()) throw std::invalid_argument("--gguf FILE is required");
    llm::gguf::File f(path);
    if (list) {
        for (const auto& [k, v] : f.meta())
            if (v.strs.empty() && v.nums.empty() && v.str.size() < 80)
                std::printf("%-45s %s\n", k.c_str(), v.str.empty() ? std::to_string(v.num).c_str() : v.str.c_str());
        std::map<std::string, int> counts;
        for (const auto& [n, t] : f.tensors())
            counts[(n.rfind("blk.", 0) == 0 ? n.substr(n.find('.', 4) + 1) : n) + " " + llm::gguf::typeName(t.type)]++;
        for (const auto& [k, c] : counts) std::printf("  %-32s x%d\n", k.c_str(), c);
        return 0;
    }
    if (ref_path.empty()) throw std::invalid_argument("--ref FILE is required");
    const Ref ref = readRef(ref_path);
    const int P = int(ref.prompt.size()), G = int(ref.gen.size());

    if (pcudaInit() != pcuda::Error::Success) throw std::runtime_error("pcudaInit failed");
    if (llm::registerKernels() != pcuda::Error::Success)
        throw std::runtime_error(std::string("registering kernels: ") + pcudaGetLastErrorDetail());
    pcuda::DeviceProperties props{};
    pcudaGetDeviceProperties(&props, 0);
    if (!workers) workers = props.worker_count + 1;

    int rc = 0;
    {
        llm::PcudaAllocator pa;
        const auto t0 = Clock::now();
        const llm::Model m = llm::loadGguf(f, P + G + 1, pa);
        const llm::Config& c = m.cfg;
        const auto vocab = llm::ggufVocab(f);
        const llm::Acts a = llm::allocActs(c, pa, 1);
        const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("Model : %s, %d layers, hidden %d, FFN %d, %d heads (%d KV), vocab %d\n", c.name.c_str(), c.layers,
                    c.hidden, c.inter, c.heads, c.kv_heads, c.vocab);
        std::printf("Weights: %.2f GB of Q4_K/Q6_K read per token (as in the file); loaded and repacked in %.1f s\n",
                    double(c.weightBytes()) / 1e9, load_s);
        std::printf("Run   : prompt %d tokens, %d generated by llama.cpp (greedy); pcuda pool %u threads + 1, graphs %s\n\n",
                    P, G, props.worker_count, graphs ? "on" : "off");
        auto pcu = llm::makePcudaBackend(m, workers, graphs);
        auto cpp = llm::makeCppBackend(m);
        auto logits = [&] {
            std::vector<float> l(size_t(c.vocab));
            pa.download(l.data(), a.logits, l.size() * sizeof(float));
            return l;
        };

        // 1. pcuda against the C++ mirror of the kernels (same K/V cache slots, same values).
        bool mirror_ok = true;
        for (int p = 0; p < std::min(mirror, P); ++p) {
            llm::decodeToken(*pcu, m, a, p, nullptr, ref.prompt[size_t(p)]);
            const auto lp = logits();
            llm::decodeToken(*cpp, m, a, p, nullptr, ref.prompt[size_t(p)]);
            const auto lc = logits();
            if (lp != lc) mirror_ok = false;
        }
        std::printf("1. pcuda vs C++ mirror, first %d positions: %s\n", std::min(mirror, P),
                    mirror_ok ? "bit-identical logits" : "DIFFERENT");

        // 2. Teacher forcing on llama.cpp's tokens.
        std::vector<int> seq = ref.prompt;
        for (const RefToken& t : ref.gen) seq.push_back(t.id);
        int agree = 0, top5_hits = 0, top5_total = 0;
        double sum_abs = 0, max_abs = 0;
        for (int p = 0; p + 1 < int(seq.size()); ++p) {
            llm::decodeToken(*pcu, m, a, p, nullptr, seq[size_t(p)]);
            if (p < P - 1) continue;
            const auto l = logits();
            const auto ls = logSoftmax(l);
            const RefToken& rt = ref.gen[size_t(p - (P - 1))];
            agree += argmax(l) == rt.id;
            const double d = std::abs(ls[size_t(rt.id)] - rt.logprob);
            sum_abs += d, max_abs = std::max(max_abs, d);
            std::vector<int> ours(l.size());
            for (size_t i = 0; i < ours.size(); ++i) ours[i] = int(i);
            const size_t kk = std::min<size_t>(rt.top.size(), 5);
            std::partial_sort(ours.begin(), ours.begin() + ptrdiff_t(kk), ours.end(),
                              [&](int x, int y) { return l[size_t(x)] > l[size_t(y)]; });
            for (size_t i = 0; i < kk; ++i)
                top5_hits += std::count(ours.begin(), ours.begin() + ptrdiff_t(kk), rt.top[i].first) > 0;
            top5_total += int(kk);
        }
        std::printf("2. teacher forcing on llama.cpp's %d tokens:\n", G);
        std::printf("   same top token %d/%d; |log-prob difference| of llama.cpp's token: mean %.4f, max %.4f; "
                    "top-5 overlap %.1f%%\n",
                    agree, G, sum_abs / G, max_abs, 100.0 * top5_hits / std::max(1, top5_total));

        // 3. Free run: greedy from the prompt, timed.
        for (int p = 0; p + 1 < P; ++p) llm::decodeToken(*pcu, m, a, p, nullptr, ref.prompt[size_t(p)]);
        std::vector<int> ours;
        int tok = ref.prompt.back();
        const auto ts = Clock::now();
        for (int i = 0; i < G; ++i) {
            llm::decodeToken(*pcu, m, a, P - 1 + i, nullptr, tok);
            tok = argmax(logits());
            ours.push_back(tok);
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - ts).count() / G;
        int same = 0;
        while (same < G && ours[size_t(same)] == ref.gen[size_t(same)].id) ++same;
        std::vector<int> theirs;
        for (const RefToken& t : ref.gen) theirs.push_back(t.id);
        std::printf("3. free run, greedy: %d of %d tokens identical to llama.cpp before the first difference\n", same, G);
        std::printf("   llama.cpp: %s\n", text(vocab, theirs).c_str());
        std::printf("   pcuda    : %s\n", text(vocab, ours).c_str());
        std::printf("   pcuda decode: %.2f ms/token, %.2f tok/s, %.1f GB/s of weights\n\n", ms, 1000.0 / ms,
                    double(c.weightBytes()) / 1e9 / (ms / 1000.0));
        std::printf("RESULT_REAL,%d,%d,%d,%.5f,%.5f,%.2f,%d,%.4f,%.4f,%.4f\n", mirror_ok ? 1 : 0, agree, G, sum_abs / G,
                    max_abs, 100.0 * top5_hits / std::max(1, top5_total), same, ms, 1000.0 / ms,
                    double(c.weightBytes()) / 1e9);
        rc = mirror_ok && agree >= G * 9 / 10 ? 0 : 1;
    }
    pcudaShutdown();
    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pcuda_llm_real: %s\n", e.what());
        return 1;
    }
}
