// kc_mslgen: turns Booth's --metal output (plus its --ir dump) into C++ for Kernelcore's native
// code paths. Booth's front end and optimiser produce good SSA; only its x86 backend is weak
// (no register allocation). The Metal backend prints the same SSA as structured C, which MSVC
// compiles with full optimisation.
//
//   kc_mslgen --metal k.metal --ir k.ir --kernel NAME --native k_native.cpp --simd k_simd.cpp
//
// Outputs
//   k_native.cpp  "native": the Metal text (with fixes below) in a namespace, plus a wrapper that
//                 unpacks Kernelcore LaunchArgs and runs threads 0..n-1 in order, like Booth's x86 code.
//                 Defines extern "C" const kc::NativeKernel kc_knat_NAME.
//   k_simd.cpp    "simd": the kernel translated to run 8 CUDA threads per AVX2 register
//                 (src/cpu/msl/kc_simd.hpp), or a stub with the reason when a construct is
//                 not supported. Uniform values (identical in every lane) stay scalar.
//
// Fixes applied from the IR, because the Metal text loses information:
//   - byte loads are printed as signed `char` and converted with a plain cast; the IR says
//     whether the conversion is zext/sext (uitofp/sitofp), so the cast is made explicit;
//   - unsigned IR operations (lshr, udiv, urem, icmp ult/...) get explicit unsigned casts in
//     the native path and disable the simd path.
// Anything the generator cannot reproduce exactly fails the build (native) or falls back
// (simd); it never guesses.

#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != '\r') out.push_back(c);
    return out;
}

void writeFile(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) die("cannot write " + path);
    f << text;
}

std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) v.push_back(line);
    return v;
}

// Booth's Metal backend prints negative i32 constants as unsigned decimals (-127 becomes
// 4294967169). In C++ such a literal is a 64-bit value, so `x < 4294967169` would always be true.
// Rewrites integer literals in [2^31, 2^32) back to their 32-bit signed value.
std::string signedLiterals(const std::string& line) {
    std::string out;
    size_t i = 0;
    auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'; };
    while (i < line.size()) {
        const bool start = std::isdigit(static_cast<unsigned char>(line[i])) && (i == 0 || !ident(line[i - 1]));
        if (!start) {
            out.push_back(line[i++]);
            continue;
        }
        size_t j = i;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) ++j;
        const std::string digits = line.substr(i, j - i);
        const bool plain = j == line.size() || !(ident(line[j]) || line[j] == 'e' || line[j] == 'E');
        if (plain && digits.size() == 10) {
            const unsigned long long v = std::stoull(digits);
            if (v >= 2147483648ull && v < 4294967296ull) {
                out += std::to_string(static_cast<long long>(v) - 4294967296ll);  // "a - -127" is valid C
                i = j;
                continue;
            }
        }
        out += digits;
        i = j;
    }
    return out;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}

// ---- IR facts -----------------------------------------------------------------

struct IrInst {
    std::string op;    // zext, sext, sitofp, ashr, lshr, sdiv, icmp, fcmp, ...
    std::string pred;  // icmp/fcmp predicate
    std::string from;  // source type of a cast
};

std::map<int, IrInst> parseIr(const std::string& text, const std::string& kernel) {
    std::map<int, IrInst> out;
    bool in = false;
    const std::regex fn(R"(^func @(\w+)\()");
    const std::regex inst(R"(^\s*%(\d+) = (\w+)\s*(.*)$)");
    for (const std::string& line : splitLines(text)) {
        std::smatch m;
        if (std::regex_search(line, m, fn)) {
            in = m[1] == kernel;
            continue;
        }
        if (!in || !std::regex_match(line, m, inst)) continue;
        IrInst ii;
        ii.op = m[2];
        std::stringstream rest(m[3].str());
        if (ii.op == "icmp" || ii.op == "fcmp") rest >> ii.pred;
        else if (ii.op == "zext" || ii.op == "sext" || ii.op == "sitofp" || ii.op == "uitofp" || ii.op == "fptosi" ||
                 ii.op == "fptoui" || ii.op == "trunc")
            rest >> ii.from;
        out[std::stoi(m[1])] = ii;
    }
    if (out.empty()) die("no IR found for kernel " + kernel);
    return out;
}

// ---- Metal structure ----------------------------------------------------------

enum class Ty { Int, Float, Bool, Char, PtrFloat, PtrInt, PtrChar };

bool isPtr(Ty t) { return t == Ty::PtrFloat || t == Ty::PtrInt || t == Ty::PtrChar; }
Ty elemOf(Ty t) { return t == Ty::PtrFloat ? Ty::Float : t == Ty::PtrInt ? Ty::Int : Ty::Char; }
const char* cElem(Ty t) { return t == Ty::PtrFloat ? "float" : t == Ty::PtrInt ? "int" : "char"; }

Ty parseTy(const std::string& s) {
    if (s == "int") return Ty::Int;
    if (s == "float") return Ty::Float;
    if (s == "bool") return Ty::Bool;
    if (s == "char") return Ty::Char;
    if (s == "device float *") return Ty::PtrFloat;
    if (s == "device int *") return Ty::PtrInt;
    if (s == "device char *") return Ty::PtrChar;
    die("unsupported Metal type '" + s + "'");
}

struct Param {
    std::string name;  // p0.. or tid/bid/bdim
    Ty          ty = Ty::Int;
    bool        builtin = false;
};

struct Kernel {
    std::vector<Param>        params;
    std::map<std::string, Ty> vars;      // declared locals
    std::vector<std::string>  body;      // trimmed lines between the braces (no declarations)
    bool                      helpers = false;  // other functions in the file
    // Helper function bodies (statements only, declarations dropped), by name.
    std::map<std::string, std::vector<std::string>> helper_bodies;
    size_t                    kernel_line = 0;
};

Kernel parseMetal(const std::vector<std::string>& lines, const std::string& name) {
    Kernel k;
    const std::regex fn_def(R"(^[A-Za-z_][\w ]*\b(\w+)\($)");
    size_t i = 0;
    const std::regex local_decl(R"(^(int|float|bool|char|device float \*|device int \*|device char \*) v\d+;$)");
    for (; i < lines.size(); ++i) {
        if (lines[i] == "kernel void " + name + "(") break;
        std::smatch m;
        if (!std::regex_match(lines[i], m, fn_def)) continue;
        k.helpers = true;
        const std::string fname = m[1];
        size_t j = i + 1;
        while (j < lines.size() && trim(lines[j]).back() != ')' && trim(lines[j]).back() != ';') ++j;
        if (j + 1 >= lines.size() || trim(lines[j]).back() == ';' || lines[j + 1] != "{") continue;  // prototype
        std::vector<std::string> body;
        for (j += 2; j < lines.size() && lines[j] != "}"; ++j) {
            const std::string t = trim(lines[j]);
            if (!std::regex_match(t, local_decl)) body.push_back(t);
        }
        k.helper_bodies[fname] = body;
        i = j;
    }
    if (i == lines.size()) die("kernel '" + name + "' not found in Metal output");
    k.kernel_line = i;
    const std::regex dev(R"(^device (float|int|char) \* (p\d+) \[\[buffer\(\d+\)\]\][,)]$)");
    const std::regex con(R"(^constant (int|float) &(p\d+) \[\[buffer\(\d+\)\]\][,)]$)");
    const std::regex bi(R"(^uint3 (tid|bid|bdim) \[\[\w+\]\][,)]$)");
    for (++i; i < lines.size(); ++i) {
        const std::string t = trim(lines[i]);
        std::smatch m;
        if (std::regex_match(t, m, dev)) k.params.push_back({m[2], parseTy("device " + m[1].str() + " *")});
        else if (std::regex_match(t, m, con)) k.params.push_back({m[2], parseTy(m[1])});
        else if (std::regex_match(t, m, bi)) k.params.push_back({m[1], Ty::Int, true});
        else die("unsupported kernel parameter: " + t);
        if (t.back() == ')') break;
    }
    if (++i >= lines.size() || lines[i] != "{") die("kernel body not found");
    const std::regex decl(R"(^(int|float|bool|char|device float \*|device int \*|device char \*) (v\d+);$)");
    for (++i; i < lines.size() && lines[i] != "}"; ++i) {
        const std::string t = trim(lines[i]);
        std::smatch m;
        if (std::regex_match(t, m, decl)) k.vars[m[2]] = parseTy(m[1]);
        else k.body.push_back(t);
    }
    if (i == lines.size()) die("unterminated kernel body");
    return k;
}

// ---- native (scalar) path -----------------------------------------------------

// Rewrites lines whose meaning the Metal text gets wrong or leaves ambiguous; returns the
// fixed text and, through `simd_block`, the first reason the simd path cannot be used.
std::string fixMetal(const std::vector<std::string>& lines, const Kernel& k, const std::map<int, IrInst>& ir,
                     std::string& simd_block) {
    const std::regex cast(R"(^(\s*)(v(\d+)) = \((int|float)\)\((v\d+)\);$)");
    const std::regex bin(R"(^(\s*)(v(\d+)) = (\S+) (>>|/|%|<|<=|>|>=|==|!=) (\S+);$)");
    auto tyOf = [&](const std::string& v) -> const Ty* {
        auto it = k.vars.find(v);
        return it == k.vars.end() ? nullptr : &it->second;
    };
    // Qualify math calls so the CRT's own sin/exp2/... overloads cannot be picked instead.
    static const std::regex mathcall(R"(\b(sin|cos|exp2|log2|sqrt|fmax|fmin)\()");
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string line = lines[i];
        if (line.rfind("#include", 0) != 0) line = std::regex_replace(line, mathcall, "metal::$1(");
        std::smatch m;
        if (i > k.kernel_line && std::regex_match(line, m, cast)) {
            const Ty* src = tyOf(m[5]);
            if (src && *src == Ty::Char) {
                auto it = ir.find(std::stoi(m[3]));
                if (it == ir.end()) die("no IR instruction for " + m[2].str() + "; cannot tell byte signedness");
                const std::string& op = it->second.op;
                const char* via = op == "zext" || op == "uitofp" ? "unsigned char" :
                                  op == "sext" || op == "sitofp" ? "signed char" : nullptr;
                if (!via) die("unexpected IR '" + op + "' for byte conversion " + m[2].str());
                line = m[1].str() + m[2].str() + " = (" + m[4].str() + ")(" + via + ")(" + m[5].str() + ");";
            } else if (auto it = ir.find(std::stoi(m[3])); it != ir.end() &&
                                                           (it->second.op == "uitofp" || it->second.op == "fptoui")) {
                line = m[1].str() + m[2].str() + " = (" + m[4].str() + ")(unsigned)(" + m[5].str() + ");";
                if (simd_block.empty()) simd_block = "unsigned conversion (" + it->second.op + ")";
            }
        } else if (i > k.kernel_line && std::regex_match(line, m, bin)) {
            auto it = ir.find(std::stoi(m[3]));
            if (it != ir.end()) {
                const IrInst& ii = it->second;
                const bool uns = ii.op == "lshr" || ii.op == "udiv" || ii.op == "urem" ||
                                 (ii.op == "icmp" && ii.pred.size() > 1 && ii.pred[0] == 'u');
                if (ii.op == "fcmp" && ii.pred != "olt" && ii.pred != "ole" && ii.pred != "ogt" && ii.pred != "oge" &&
                    ii.pred != "oeq" && ii.pred != "une")
                    die("unsupported float comparison '" + ii.pred + "' in " + m[2].str());
                if (uns) {
                    line = m[1].str() + m[2].str() + " = (unsigned)(" + m[4].str() + ") " + m[5].str() + " (unsigned)(" +
                           m[6].str() + ");";
                    if (simd_block.empty()) simd_block = "unsigned integer operation (" + ii.op + " " + ii.pred + ")";
                }
            }
        }
        out += line + "\n";
    }
    return out;
}

std::string unpackParams(const Kernel& k, std::string& call_args) {
    std::string s;
    int gi = 0, xi = 0;
    for (const Param& p : k.params) {
        if (p.builtin) continue;
        if (isPtr(p.ty)) {
            s += "    " + std::string(cElem(p.ty)) + "* const " + p.name + " = reinterpret_cast<" + cElem(p.ty) +
                 "*>(a->gpr[" + std::to_string(gi++) + "]);\n";
        } else if (p.ty == Ty::Int) {
            s += "    int " + p.name + " = static_cast<int>(static_cast<int64_t>(a->gpr[" + std::to_string(gi++) +
                 "]));\n";
        } else {
            s += "    float " + p.name + " = std::bit_cast<float>(static_cast<uint32_t>(a->xmm[" +
                 std::to_string(xi++) + "]));\n";
        }
    }
    if (gi > 5) die("more than 5 integer/pointer parameters (Booth's limit)");
    (void)call_args;
    return s;
}

std::string nativeFile(const std::string& name, const std::string& fixed, const Kernel& k, bool simd,
                       const std::string& simd_note) {
    std::string call;
    for (const Param& p : k.params) {
        if (!call.empty()) call += ", ";
        if (!p.builtin) call += p.name;
        else if (p.name == "tid") call += "metal::uint3{t, 0, 0}";
        else if (p.name == "bid") call += "metal::uint3{0, 0, 0}";
        else call += "metal::uint3{n, 1, 1}";
    }
    std::string dummy;
    std::string s = "// Generated by kc_mslgen from Booth's --metal output for '" + name + "'. Do not edit.\n"
                    "#include <kernelcore/kernel_registry.hpp>\n"
                    "#include <bit>\n#include <cstdint>\n"
                    "#include <metal_stdlib>  // Kernelcore stand-in: src/cpu/msl/metal_stdlib\n"
                    "#pragma warning(disable : 4100 4189 4244 4245 4305 4700)  // generated three-address code\n\n"
                    "namespace kc_msl_" + name + " {\n" + fixed + "} // namespace\n\n"
                    "namespace {\n"
                    "// Booth semantics: one block; blockIdx = 0, blockDim = nthreads, threads run in order.\n"
                    "void runScalar(const kc::LaunchArgs* a) {\n" +
                    unpackParams(k, dummy) +
                    "    const unsigned n = static_cast<unsigned>(a->nthreads);\n"
                    "    for (unsigned t = 0; t < n; ++t) kc_msl_" + name + "::" + name + "(" + call + ");\n"
                    "}\n} // namespace\n\n";
    if (simd) s += "extern \"C\" void kc_ksimd_" + name + "(const kc::LaunchArgs* a);\n";
    std::string note = simd ? "simd: 8 threads per AVX2 register" : "simd unavailable: " + simd_note;
    for (char& c : note)
        if (c == '"' || c == '\\') c = '\'';
    s += "extern \"C\" const kc::NativeKernel kc_knat_" + name + " = {&runScalar, " +
         (simd ? "&kc_ksimd_" + name : std::string("nullptr")) + ", \"" + note + "\"};\n";
    return s;
}

// ---- simd path ----------------------------------------------------------------

// Booth's Metal output for the kc_dp4a helper in kernels/llm/matvec_q4x8_q8.cu (generic path:
// u8 x s8 products of 4 packed bytes plus c). The simd path replaces calls to a helper with
// exactly this body by AVX-VNNI vpdpbusd, which computes the same integer result.
const std::vector<std::string>& dp4aCanonicalBody() {
    static const std::vector<std::string> body{
        "v3 = p0 & 255;",  "v4 = p1 << 24;",  "v5 = v4 >> 24;",   "v6 = v3 * v5;",    "v7 = p2 + v6;",
        "v8 = p0 >> 8;",   "v9 = v8 & 255;",  "v10 = p1 << 16;",  "v11 = v10 >> 24;", "v12 = v9 * v11;",
        "v13 = v7 + v12;", "v14 = p0 >> 16;", "v15 = v14 & 255;", "v16 = p1 << 8;",   "v17 = v16 >> 24;",
        "v18 = v15 * v17;", "v19 = v13 + v18;", "v20 = p0 >> 24;", "v21 = v20 & 255;", "v22 = p1 >> 24;",
        "v23 = v21 * v22;", "v24 = v19 + v23;", "return v24;"};
    return body;
}

// Booth's Metal output for the kc_h2f helper in kernels/llm/fa_scores.cu and fa_values.cu (fp16
// bits -> float with integer and float arithmetic). The simd path replaces calls to a helper with
// exactly this body by F16C vcvtph2ps: the same value for every finite fp16 (it also keeps
// infinities and NaN, which the arithmetic turns into large finite values).
const std::vector<std::string>& h2fCanonicalBody() {
    static const std::vector<std::string> body{
        "v3 = p0 >> 10;", "v4 = v3 & 31;", "v5 = v4 + 31;", "v6 = v5 >> 5;",
        "v7 = v4 + 1;", "v8 = v7 - v6;", "v9 = v8 + 1;", "v10 = v9 >> 5;",
        "v11 = v8 - v10;", "v12 = v11 & 1;", "v13 = (float)(v12);", "v14 = 1.0f + v13;",
        "v15 = v11 >> 1;", "v16 = v15 & 1;", "v17 = (float)(v16);", "v18 = 3.0f * v17;",
        "v19 = 1.0f + v18;", "v20 = v14 * v19;", "v21 = v11 >> 2;", "v22 = v21 & 1;",
        "v23 = (float)(v22);", "v24 = 15.0f * v23;", "v25 = 1.0f + v24;", "v26 = v20 * v25;",
        "v27 = v11 >> 3;", "v28 = v27 & 1;", "v29 = (float)(v28);", "v30 = 255.0f * v29;",
        "v31 = 1.0f + v30;", "v32 = v26 * v31;", "v33 = v11 >> 4;", "v34 = v33 & 1;",
        "v35 = (float)(v34);", "v36 = 65535.0f * v35;", "v37 = 1.0f + v36;", "v38 = v32 * v37;",
        "v39 = p0 & 1023;", "v40 = v6 << 10;", "v41 = v39 + v40;", "v42 = (float)(v41);",
        "v43 = v42 * v38;", "v44 = v43 * 2.98023224e-08f;", "v45 = p0 >> 15;", "v46 = v45 & 1;",
        "v47 = 2 * v46;", "v48 = 1 - v47;", "v49 = (float)(v48);", "v50 = v44 * v49;",
        "return v50;"};
    return body;
}

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void unsupported(const std::string& why) { throw Unsupported(why); }

struct Expr {
    enum Kind { Atom, Bin, Load, Cast, Call, Gep, Builtin, Select } kind = Atom;  // Select: args {cond, a, b}
    std::string              op;    // operator, function, cast type, builtin name
    std::vector<std::string> args;  // atoms
};

struct Stmt {
    enum Kind { Assign, Store, If, Loop } kind = Assign;
    int                       region = 0;
    std::string               lhs;   // Assign: variable; Store: pointer atom
    Expr                      e;     // Assign rhs; Store: e.args[0] is the value atom
    std::string               cond;  // If / Loop condition variable
    std::vector<Stmt>         a, b;  // If: then/else; Loop: header/body
    int                       ra = 0, rb = 0;  // regions of a / b
};

struct Region {
    int         parent = -1;
    std::string cond;  // condition variable ("" for the top region)
};

class Translator {
public:
    Translator(const Kernel& k, const std::map<int, IrInst>& ir, std::string name)
        : k_(k), ir_(ir), name_(std::move(name)) {
        for (const auto& p : k.params) types_[p.name] = p.ty;
        for (const auto& [v, t] : k.vars) types_[v] = t;
    }

    std::string run() {
        // Helper functions are only allowed when they are known intrinsics with the exact body
        // Kernelcore expects; anything else stays on the native path.
        for (const auto& [fname, body] : k_.helper_bodies) {
            if (fname == "kc_aligned8" && body == std::vector<std::string>{"return p0;"}) continue;
            if (fname == "kc_h2f" && body == h2fCanonicalBody()) continue;
            if (fname != "kc_dp4a" || body != dp4aCanonicalBody())
                unsupported("calls helper function '" + fname + "'");
        }
        if (k_.helpers && k_.helper_bodies.empty()) unsupported("calls a helper function");
        regions_.push_back({});
        size_t i = 0;
        std::vector<Stmt> top = parseBlock(i, 0);
        if (i != k_.body.size()) unsupported("unexpected statement '" + k_.body[i] + "'");
        top_ = std::move(top);
        collect(top_);
        analyse();
        return emit();
    }

private:
    // ---------- parsing ----------
    bool isLiteral(const std::string& a) const { return !a.empty() && (std::isdigit(uint8_t(a[0])) || a[0] == '-'); }
    bool isFloatLit(const std::string& a) const {
        return isLiteral(a) && (a.find('.') != std::string::npos || a.back() == 'f' || a.find('e') != std::string::npos);
    }
    Ty atomTy(const std::string& a) const {
        if (isLiteral(a)) return isFloatLit(a) ? Ty::Float : Ty::Int;
        auto it = types_.find(a);
        if (it == types_.end()) unsupported("unknown name '" + a + "'");
        return it->second;
    }

    Expr parseRhs(const std::string& lhs, const std::string& r) {
        static const std::regex load(R"(^\*\((\w+)\)$)");
        static const std::regex cast(R"(^\((int|float)\)\(([\w.\-]+)\)$)");
        static const std::regex call(R"(^(\w+)\(([^()]*)\)$)");
        static const std::regex gep(R"(^(\w+) \+ \(?([\w.\-]+)\)?$)");
        static const std::regex builtin(R"(^(tid|bid|bdim)\.x$)");
        static const std::regex bin(R"(^(\S+) (\+|-|\*|/|%|<|<=|>|>=|==|!=|&|\||\^|<<|>>|&&|\|\|) (\S+)$)");
        static const std::regex atom(R"(^[\w.\-]+$)");
        static const std::regex select(R"(^(\S+) \? (\S+) : (\S+)$)");
        static const std::regex ptrcast(R"(^as_type<device (float|int|char) \*>\((\w+)\)$)");
        std::smatch m;
        Expr e;
        if (std::regex_match(r, m, ptrcast)) e = {Expr::Cast, "ptr:" + m[1].str(), {m[2]}};
        else if (std::regex_match(r, m, load)) e = {Expr::Load, "", {m[1]}};
        else if (std::regex_match(r, m, builtin)) e = {Expr::Builtin, m[1], {}};
        else if (std::regex_match(r, m, cast)) e = {Expr::Cast, m[1], {m[2]}};
        else if (std::regex_match(r, m, call)) {
            e = {Expr::Call, m[1], {}};
            std::stringstream ss(m[2].str());
            std::string a;
            while (std::getline(ss, a, ',')) e.args.push_back(trim(a));
        } else if (isPtr(types_.at(lhs)) && std::regex_match(r, m, gep)) e = {Expr::Gep, "", {m[1], m[2]}};
        else if (std::regex_match(r, m, select)) {
            if (isPtr(types_.at(lhs))) unsupported("pointer select '" + r + "'");
            e = {Expr::Select, "", {m[1], m[2], m[3]}};
        } else if (std::regex_match(r, m, bin)) e = {Expr::Bin, m[2], {m[1], m[3]}};
        else if (std::regex_match(r, m, atom)) e = {Expr::Atom, "", {r}};
        else unsupported("expression '" + r + "'");
        return e;
    }

    int newRegion(int parent, const std::string& cond) {
        regions_.push_back({parent, cond});
        return int(regions_.size()) - 1;
    }

    static bool isTerminator(const std::string& t) {
        return t == "}" || t == "} else {" || t == "continue;" || t == "break;";
    }

    // Statements until a closing brace, `} else {`, `continue;` or `break;` (not consumed).
    std::vector<Stmt> parseBlock(size_t& i, int region) {
        std::vector<Stmt> out;
        const auto& L = k_.body;
        while (i < L.size() && !isTerminator(L[i])) {
            if (L[i] == "return;") {
                ++i;
                if (region != 0 || i != L.size()) unsupported("return before the end of the kernel");
                continue;
            }
            out.push_back(parseStmt(i, region));
        }
        return out;
    }

    void expect(size_t& i, const char* want) {
        if (i >= k_.body.size() || k_.body[i] != want) unsupported(std::string("expected '") + want + "'");
        ++i;
    }

    Stmt parseStmt(size_t& i, int region) {
        static const std::regex assign(R"(^(v\d+) = (.+);$)");
        static const std::regex store(R"(^\*\((\w+)\) = ([\w.\-]+);$)");
        static const std::regex ifs(R"(^if \((\w+)\) \{$)");
        const auto& L = k_.body;
        const std::string& t = L[i];
        std::smatch m;
        Stmt s;
        s.region = region;
        if (std::regex_match(t, m, assign)) {
            s.kind = Stmt::Assign;
            s.lhs = m[1];
            if (!types_.count(s.lhs)) unsupported("undeclared " + s.lhs);
            s.e = parseRhs(s.lhs, m[2]);
            ++i;
        } else if (std::regex_match(t, m, store)) {
            s.kind = Stmt::Store;
            s.lhs = m[1];
            s.e = {Expr::Atom, "", {m[2]}};
            ++i;
        } else if (std::regex_match(t, m, ifs)) {
            s.kind = Stmt::If;
            s.cond = m[1];
            ++i;
            s.ra = newRegion(region, s.cond);
            s.a = parseBlock(i, s.ra);
            s.rb = newRegion(region, "!" + s.cond);
            if (i < L.size() && L[i] == "} else {") {
                ++i;
                s.b = parseBlock(i, s.rb);
            }
            expect(i, "}");
        } else if (t == "while (true) {") {
            // Booth's canonical loop:
            //   while (true) { header...; if (c) { body...; continue; } else { break; } }
            ++i;
            s.kind = Stmt::Loop;
            s.ra = newRegion(region, "");
            while (i < L.size() && !std::regex_match(L[i], m, ifs)) {
                if (isTerminator(L[i]) || L[i] == "while (true) {") unsupported("loop header shape");
                s.a.push_back(parseStmt(i, s.ra));
            }
            if (i >= L.size()) unsupported("loop without condition");
            s.cond = m[1];
            regions_[size_t(s.ra)].cond = s.cond;
            ++i;
            s.b = parseBlock(i, s.ra);
            expect(i, "continue;");
            expect(i, "} else {");
            expect(i, "break;");
            expect(i, "}");
            expect(i, "}");
        } else {
            unsupported("statement '" + t + "'");
        }
        return s;
    }

    // ---------- analysis ----------
    struct Site {
        int region;
    };
    std::map<std::string, std::vector<const Stmt*>> assigns_;
    std::map<std::string, std::vector<int>>         uses_;

    void use(const std::string& a, int region) {
        if (!isLiteral(a)) uses_[a].push_back(region);
    }
    void collect(const std::vector<Stmt>& blk) {
        for (const Stmt& s : blk) {
            switch (s.kind) {
            case Stmt::Assign:
                assigns_[s.lhs].push_back(&s);
                for (const auto& a : s.e.args) use(a, s.region);
                break;
            case Stmt::Store:
                use(s.lhs, s.region);
                use(s.e.args[0], s.region);
                break;
            case Stmt::If:
                use(s.cond, s.region);
                collect(s.a);
                collect(s.b);
                break;
            case Stmt::Loop:
                use(s.cond, s.ra);
                collect(s.a);
                collect(s.b);
                break;
            }
        }
    }

    bool inside(int r, int anc) const {
        for (; r >= 0; r = regions_[size_t(r)].parent)
            if (r == anc) return true;
        return false;
    }
    std::string condVar(int r) const {
        std::string c = regions_[size_t(r)].cond;
        return !c.empty() && c[0] == '!' ? c.substr(1) : c;
    }
    bool regionVarying(int r) const {  // the condition differs between lanes
        const std::string c = condVar(r);
        if (c.empty()) return false;
        const Shape s = shapeOf(c);
        return s.k == Shape::Aff || s.k == Shape::Gen;
    }
    int innermostVarying(int r) const {
        for (; r > 0; r = regions_[size_t(r)].parent)
            if (regionVarying(r)) return r;
        return -1;
    }
    bool escapes(const std::string& v, int vr) const {
        if (auto it = uses_.find(v); it != uses_.end())
            for (int u : it->second)
                if (!inside(u, vr)) return true;
        if (auto it = assigns_.find(v); it != assigns_.end())
            for (const Stmt* s : it->second)
                if (!inside(s->region, vr)) return true;
        return false;
    }

    // Shape of a value across the 8 lanes of a group:
    //   Uni  the same in every lane (scalar; `c` = its value when known at generation time)
    //   Aff  lane l holds b + stride*l: kept as the scalar b (the lane-0 value), stride constant.
    //        Int and pointer values only (pointer stride in elements).
    //   `aligned`: the (lane-0) value is a non-negative multiple of 8. For an Aff value with
    //        stride 1 the lanes b..b+7 then share one quotient x / D for any aligned divisor D,
    //        and x % D is b % D + lane. tid is aligned (groups start at multiples of 8);
    //        uniform values are aligned when constant or promised by kc_aligned8().
    //   Gen  anything else: a full vector
    // Uni and Aff values are computed by the scalar C expression on lane-0 values, which gives
    // exactly lane 0's value; other lanes follow from the stride.
    struct Shape {
        enum K { Top, Uni, Aff, Gen } k = Top;  // Top: not known yet (optimistic start)
        long long                 stride = 0;
        bool                      aligned = false;
        std::optional<long long>  c;
        bool operator==(const Shape& o) const {
            return k == o.k && stride == o.stride && aligned == o.aligned && c == o.c;
        }
    };
    static Shape uniS(std::optional<long long> c = std::nullopt, bool al = false) {
        return {Shape::Uni, 0, al || (c && *c >= 0 && *c % 8 == 0), c};
    }
    static Shape gen() { return {Shape::Gen, 0, false, std::nullopt}; }
    static Shape aff(long long s, bool al) {
        if (s == 0) return uniS();
        if (s > (1 << 20) || s < -(1 << 20)) return gen();
        return {Shape::Aff, s, al, std::nullopt};
    }
    static Shape meet(const Shape& a, const Shape& b) {
        if (a.k == Shape::Top) return b;
        if (b.k == Shape::Top) return a;
        if (a.k == Shape::Gen || b.k == Shape::Gen || a.k != b.k) return gen();
        if (a.k == Shape::Uni) return uniS(a.c == b.c ? a.c : std::nullopt, a.aligned && b.aligned);
        if (a.stride != b.stride) return gen();
        return {Shape::Aff, a.stride, a.aligned && b.aligned, std::nullopt};
    }
    Shape shapeOf(const std::string& a) const {
        if (isLiteral(a)) {
            if (isFloatLit(a)) return uniS();
            try {
                return uniS(std::stoll(a));
            } catch (...) {
                return uniS();
            }
        }
        auto it = shape_.find(a);
        return it == shape_.end() ? uniS() : it->second;  // parameters are uniform
    }
    static std::optional<long long> fold(const std::string& op, long long a, long long b) {
        long long r;
        if (op == "+") r = a + b;
        else if (op == "-") r = a - b;
        else if (op == "*") r = a * b;
        else if (op == "/" && b != 0) r = a / b;
        else if (op == "%" && b != 0) r = a % b;
        else return std::nullopt;
        if (r < INT32_MIN || r > INT32_MAX) return std::nullopt;
        return r;
    }

    Shape transfer(const Stmt& s) const {
        const Expr& e = s.e;
        const Ty lt = types_.at(s.lhs);
        Shape r;
        switch (e.kind) {
        case Expr::Atom: r = shapeOf(e.args[0]); break;
        case Expr::Builtin:
            r = e.op == "tid" ? Shape{Shape::Aff, 1, true, std::nullopt} : e.op == "bid" ? uniS(0) : uniS();
            break;
        case Expr::Bin: {
            const Shape a = shapeOf(e.args[0]), b = shapeOf(e.args[1]);
            if (a.k == Shape::Top || b.k == Shape::Top) return {};
            if (a.k == Shape::Gen || b.k == Shape::Gen) return gen();
            const bool f = atomTy(e.args[0]) == Ty::Float || atomTy(e.args[1]) == Ty::Float;
            const std::string& op = e.op;
            if (a.k == Shape::Uni && b.k == Shape::Uni) {
                if (!f && op == "*" && (a.c == 0 || b.c == 0)) r = uniS(0);  // blockIdx (0) * blockDim
                // aligned (a multiple of 8): sums and differences of two, products with one
                else r = uniS(!f && a.c && b.c ? fold(op, *a.c, *b.c) : std::nullopt,
                              !f && (((op == "+" || op == "-") && a.aligned && b.aligned) ||
                                     (op == "*" && (a.aligned || b.aligned))));
                break;
            }
            if (f) return gen();
            const bool ua = a.k == Shape::Uni, ub = b.k == Shape::Uni;
            if (op == "+") {
                if (!ua && !ub) r = aff(a.stride + b.stride, false);
                else {
                    const Shape& x = ua ? b : a;
                    const Shape& u = ua ? a : b;
                    r = aff(x.stride, x.aligned && u.aligned);
                }
            } else if (op == "-") {
                if (!ua && !ub) r = aff(a.stride - b.stride, false);
                else if (ub) r = aff(a.stride, a.aligned && (b.c == 0 || b.aligned));  // minus a multiple of 8
                else r = aff(-b.stride, false);
            } else if (op == "*") {
                const Shape& x = ua ? b : a;
                const Shape& u = ua ? a : b;
                if (!ua && !ub) r = gen();
                else if (!u.c) r = gen();
                else if (*u.c == 0) r = uniS(0);
                else r = aff(x.stride * *u.c, false);
            } else if (op == "/" && !ua && ub && a.stride == 1 && a.aligned && b.aligned) {
                r = uniS();  // b..b+7 lie between two multiples of D: one quotient
            } else if (op == "%" && !ua && ub && a.stride == 1 && a.aligned && b.aligned) {
                r = Shape{Shape::Aff, 1, true, std::nullopt};  // b % D + lane
            } else if (op == "<<" && !ua && ub && b.c && *b.c >= 0 && *b.c <= 20) {
                r = aff(a.stride << *b.c, false);
            } else {
                r = gen();
            }
            break;
        }
        case Expr::Select:
            r = uniS();
            for (const auto& x : e.args) {
                const Shape a = shapeOf(x);
                if (a.k == Shape::Top) return {};
                if (a.k != Shape::Uni) r = gen();
            }
            break;
        case Expr::Gep: {
            const Shape p = shapeOf(e.args[0]), i = shapeOf(e.args[1]);
            if (p.k == Shape::Top || i.k == Shape::Top) return {};
            if (p.k == Shape::Gen || i.k == Shape::Gen) return gen();
            r = aff(p.stride + i.stride, false);
            break;
        }
        case Expr::Cast:
            r = shapeOf(e.args[0]);
            if (r.k == Shape::Top) return {};
            if (e.op.rfind("ptr:", 0) != 0) {  // value conversion
                if (r.k == Shape::Aff) r = gen();
                else if (r.k == Shape::Uni && !(e.op == "int" && atomTy(e.args[0]) == Ty::Int)) r = uniS();
            }
            break;
        case Expr::Load: {
            const Shape p = shapeOf(e.args[0]);
            if (p.k == Shape::Top) return {};
            r = p.k == Shape::Uni ? uniS() : gen();
            break;
        }
        case Expr::Call:
            if (e.op == "kc_aligned8" && e.args.size() == 1) {  // checked promise, identity
                r = shapeOf(e.args[0]);
                if (r.k == Shape::Top) return {};
                if (r.k == Shape::Uni) r.aligned = true;
                break;
            }
            r = uniS();
            for (const auto& x : e.args) {
                const Shape a = shapeOf(x);
                if (a.k == Shape::Top) return {};
                if (a.k != Shape::Uni) r = gen();
            }
            break;
        }
        if (r.k == Shape::Aff && lt != Ty::Int && !isPtr(lt)) r = gen();
        if (r.k == Shape::Uni && lt != Ty::Int) r.c.reset(), r.aligned = false;
        return r;
    }

    void analyse() {
        // Optimistic fixed point: shapes only move down (Top -> Uni/Aff -> Gen).
        for (const auto& [v, list] : assigns_) shape_[v] = {};
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& [v, list] : assigns_) {
                Shape& cur = shape_[v];
                if (cur.k == Shape::Gen) continue;
                Shape n;
                for (const Stmt* s : list) {
                    n = meet(n, transfer(*s));
                    // Assigned under a partial mask and observed where the mask differs: blended.
                    const int vr = innermostVarying(s->region);
                    if (vr > 0 && escapes(v, vr)) n = gen();
                }
                n = meet(cur, n);
                // A vector pointer derived from an affine pointer keeps one base for blends:
                // only uniform pointers serve as bases, so such a pointer becomes a vector too.
                if (n.k == Shape::Aff && isPtr(types_.at(v)))
                    for (const auto& [w, wl] : assigns_)
                        if (w != v && shape_[w].k == Shape::Gen && isPtr(types_.at(w)))
                            for (const Stmt* s : wl)
                                if (s->kind == Stmt::Assign && !s->e.args.empty() && s->e.args[0] == v) n = gen();
                if (!(n == cur)) {
                    cur = n;
                    changed = true;
                }
            }
        }
        for (auto& [v, sh] : shape_) {
            if (sh.k == Shape::Top) sh = gen();  // never resolved: stay safe
            if (sh.k == Shape::Gen) varying_.insert(v);
            else if (sh.k == Shape::Aff) aff_[v] = int(sh.stride);
        }
        // Blend when a value assigned under a partial mask is observed (or reassigned) where the
        // mask differs: outside the innermost varying region of the assignment.
        for (const auto& [v, list] : assigns_) {
            if (!varying_.count(v)) continue;
            for (const Stmt* s : list) {
                const int vr = innermostVarying(s->region);
                if (vr > 0 && escapes(v, vr)) blend_.insert(v);
            }
        }
        // Varying pointers must keep one base, since blends only merge the index.
        for (const auto& [v, list] : assigns_) {
            if (!varying_.count(v) || !isPtr(types_.at(v))) continue;
            std::string base;
            for (const Stmt* s : list) {
                std::string b;
                if (s->e.kind == Expr::Gep || s->e.kind == Expr::Atom || s->e.kind == Expr::Cast)
                    b = baseOf(s->e.args[0]);
                else unsupported("varying pointer from '" + s->e.op + "'");
                if (!base.empty() && b != base) unsupported("varying pointer " + v + " with different bases");
                base = b;
            }
            ptr_base_[v] = base;
        }
    }

    std::string baseOf(const std::string& p) const {
        if (auto it = ptr_base_.find(p); it != ptr_base_.end()) return it->second;
        if (varying_.count(p)) {
            // Resolve through its own assignments.
            auto it = assigns_.find(p);
            if (it == assigns_.end()) unsupported("pointer " + p + " has no assignment");
            const Stmt* s = it->second.front();
            if (s->e.kind == Expr::Gep || s->e.kind == Expr::Atom || s->e.kind == Expr::Cast)
                return baseOf(s->e.args[0]);
            unsupported("pointer " + p);
        }
        return p;  // uniform pointer: itself
    }

    // ---------- emission ----------
    std::string uni(const std::string& a) const {
        if (isLiteral(a)) return a;
        return a;
    }
    // Operand as a vector of type t.
    std::string vec(const std::string& a, Ty t) const {
        if (!isLiteral(a) && varying_.count(a)) return a;
        if (auto it = aff_.find(a); it != aff_.end()) {
            const std::string v = "affine(" + a + ", " + std::to_string(it->second) + ")";
            return t == Ty::Float ? "i2f(" + v + ")" : v;
        }
        if (t == Ty::Float) return "splat(static_cast<float>(" + a + "))";
        return "splat(static_cast<int>(" + a + "))";
    }
    Ty scalarKind(Ty t) const { return t == Ty::Float ? Ty::Float : Ty::Int; }  // bool/char lanes are ints

    std::string vexpr(const Stmt& s) const {
        const Expr& e = s.e;
        const Ty lt = types_.at(s.lhs);
        switch (e.kind) {
        case Expr::Atom:
            if (isPtr(lt)) return e.args[0];
            return vec(e.args[0], scalarKind(lt));
        case Expr::Builtin:
            if (e.op == "tid") return "add(splat(static_cast<int>(t0_)), iota())";
            unsupported("varying " + e.op);
        case Expr::Bin: {
            const Ty ta = atomTy(e.args[0]), tb = atomTy(e.args[1]);
            const bool f = ta == Ty::Float || tb == Ty::Float;
            const Ty k = f ? Ty::Float : Ty::Int;
            const std::string a = vec(e.args[0], k), b = vec(e.args[1], k);
            const std::string& op = e.op;
            if (op == "+") return "add(" + a + ", " + b + ")";
            if (op == "-") return "sub(" + a + ", " + b + ")";
            if (op == "*") return "mul(" + a + ", " + b + ")";
            if (op == "/") return f ? "fdiv(" + a + ", " + b + ")" : "div(" + a + ", " + b + ", " + mask_ + ")";
            if (op == "%") {
                if (f) unsupported("float %");
                return "rem(" + a + ", " + b + ", " + mask_ + ")";
            }
            if (op == "<") return "lt(" + a + ", " + b + ")";
            if (op == "<=") return "le(" + a + ", " + b + ")";
            if (op == ">") return "gt(" + a + ", " + b + ")";
            if (op == ">=") return "ge(" + a + ", " + b + ")";
            if (op == "==") return "eq(" + a + ", " + b + ")";
            if (op == "!=") return "ne(" + a + ", " + b + ")";
            if (f) unsupported("float operator " + op);
            if (op == "&" || op == "&&") return "band(" + a + ", " + b + ")";
            if (op == "|" || op == "||") return "bor(" + a + ", " + b + ")";
            if (op == "^") return "bxor(" + a + ", " + b + ")";
            if (op == "<<") return "shl(" + a + ", " + b + ")";
            if (op == ">>") return "ashr(" + a + ", " + b + ")";
            unsupported("operator " + op);
        }
        case Expr::Select: {
            const Ty k = scalarKind(lt);
            return "blend(" + vec(e.args[2], k) + ", " + vec(e.args[1], k) + ", ne(" + vec(e.args[0], Ty::Int) + ", zeroI()))";
        }
        case Expr::Load: {
            const std::string& p = e.args[0];
            const Ty pt = atomTy(p);
            if (auto it = aff_.find(p); it != aff_.end())  // lane l reads p[stride * l]
                return it->second == 1 ? "loadc(" + p + ", " + mask_ + ")"
                                       : "load(affinePtr(" + p + ", " + std::to_string(it->second) + "), " + mask_ + ")";
            if (!varying_.count(p)) return vec("(*" + p + ")", scalarKind(elemOf(pt)));
            return "load(" + p + ", " + mask_ + ")";
        }
        case Expr::Cast: {
            const std::string& x = e.args[0];
            const Ty xt = atomTy(x);
            if (e.op.rfind("ptr:", 0) == 0) {  // as_type pointer cast
                if (!isPtr(xt)) unsupported("as_type of a non-pointer");
                const std::string t = e.op.substr(4);
                if (!varying_.count(x)) return "VPtr<" + t + ">{reinterpret_cast<" + t + "*>(" + x + "), zeroI()}";
                return "ptrcast<" + t + ">(" + x + ")";
            }
            std::string v = vec(x, scalarKind(xt));
            if (xt == Ty::Char) v = byteExt(s, v);
            if (e.op == "float") return xt == Ty::Float ? v : "i2f(" + v + ")";
            return xt == Ty::Float ? "f2i(" + v + ")" : v;
        }
        case Expr::Call: {
            if (e.op == "kc_aligned8" && e.args.size() == 1) return vec(e.args[0], Ty::Int);  // not uniform: unchecked
            if (e.op == "kc_h2f" && e.args.size() == 1)  // verified helper: F16C vcvtph2ps
                return "vh2f(" + vec(e.args[0], Ty::Int) + ")";
            if (e.op == "kc_dp4a" && e.args.size() == 3)  // verified helper: AVX-VNNI vpdpbusd
                return "vdp4aT<kVnni_>(" + vec(e.args[2], Ty::Int) + ", " + vec(e.args[0], Ty::Int) + ", " +
                       vec(e.args[1], Ty::Int) + ")";
            std::vector<std::string> a;
            for (const auto& x : e.args) a.push_back(vec(x, Ty::Float));
            if (e.op == "sin" && a.size() == 1) return "vsin(" + a[0] + ")";
            if (e.op == "cos" && a.size() == 1) return "vcos(" + a[0] + ")";
            if (e.op == "exp2" && a.size() == 1) return "vexp2(" + a[0] + ")";
            if (e.op == "log2" && a.size() == 1) return "vlog2(" + a[0] + ")";
            if (e.op == "sqrt" && a.size() == 1) return "vsqrt(" + a[0] + ")";
            if (e.op == "fmax" && a.size() == 2) return "vfmax(" + a[0] + ", " + a[1] + ")";
            if (e.op == "fmin" && a.size() == 2) return "vfmin(" + a[0] + ", " + a[1] + ")";
            unsupported("call " + e.op);
        }
        case Expr::Gep: {
            const std::string& p = e.args[0];
            const std::string idx = vec(e.args[1], Ty::Int);
            return "gep(" + p + ", " + idx + ")";
        }
        }
        unsupported("expression");
    }

    // Byte lanes hold the raw byte; apply the IR's signedness on conversion.
    std::string byteExt(const Stmt& s, const std::string& v) const {
        const int n = std::stoi(s.lhs.substr(1));
        auto it = ir_.find(n);
        if (it == ir_.end()) unsupported("byte conversion without IR");
        const std::string& op = it->second.op;
        if (op == "zext" || op == "uitofp") return "zext8(" + v + ")";
        if (op == "sext" || op == "sitofp") return "sext8(" + v + ")";
        unsupported("byte conversion " + op);
    }

    std::string sexpr(const Stmt& s) const {  // uniform expression: plain C++ (scalar semantics)
        const Expr& e = s.e;
        switch (e.kind) {
        case Expr::Atom: return e.args[0];
        case Expr::Builtin:
            if (e.op == "bid") return "0";
            if (e.op == "bdim") return "static_cast<int>(n_)";
            if (e.op == "tid") return "static_cast<int>(t0_)";  // affine: lane 0's thread index
            unsupported("builtin " + e.op);
        case Expr::Bin: return e.args[0] + " " + e.op + " " + e.args[1];
        case Expr::Select: return e.args[0] + " ? " + e.args[1] + " : " + e.args[2];
        case Expr::Load: return "*(" + e.args[0] + ")";
        case Expr::Cast: {
            const Ty xt = atomTy(e.args[0]);
            if (e.op.rfind("ptr:", 0) == 0) return "reinterpret_cast<" + e.op.substr(4) + "*>(" + e.args[0] + ")";
            if (xt == Ty::Char) {
                const int n = std::stoi(s.lhs.substr(1));
                auto it = ir_.find(n);
                if (it == ir_.end()) unsupported("byte conversion without IR");
                const bool z = it->second.op == "zext" || it->second.op == "uitofp";
                return "(" + e.op + ")(" + (z ? "unsigned char" : "signed char") + ")(" + e.args[0] + ")";
            }
            return "(" + e.op + ")(" + e.args[0] + ")";
        }
        case Expr::Call: {
            std::string a;
            for (const auto& x : e.args) a += (a.empty() ? "" : ", ") + x;
            if (e.op == "kc_dp4a") return "kc::simd::dp4aScalar(" + a + ")";
            if (e.op == "kc_h2f") return "kc::simd::h2fScalar(" + a + ")";
            if (e.op == "kc_aligned8")  // checked when uniform; otherwise the identity
                return shapeOf(e.args[0]).k == Shape::Uni ? "kc::simd::aligned8(" + a + ")" : a;
            static const std::set<std::string> known{"sin", "cos", "exp2", "log2", "sqrt", "fmax", "fmin"};
            if (!known.count(e.op)) unsupported("call " + e.op);
            return "metal::" + e.op + "(" + a + ")";
        }
        case Expr::Gep: return e.args[0] + " + (" + e.args[1] + ")";
        }
        unsupported("expression");
    }

    void line(const std::string& l) { out_ += std::string(size_t(indent_) * 4, ' ') + l + "\n"; }

    void emitBlock(const std::vector<Stmt>& blk) {
        for (const Stmt& s : blk) emitStmt(s);
    }

    void emitStmt(const Stmt& s) {
        switch (s.kind) {
        case Stmt::Assign: {
            // Contiguous loads inside loops stream through memory: prefetch ahead of them
            // (the hardware prefetcher stops at 4 KB page boundaries).
            if (loop_depth_ > 0 && s.e.kind == Expr::Load)
                if (auto it = aff_.find(s.e.args[0]); it != aff_.end() && it->second == 1)
                    line("prefetchAhead(" + s.e.args[0] + ");");
            if (!varying_.count(s.lhs)) {
                line(s.lhs + " = " + sexpr(s) + ";");
            } else if (blend_.count(s.lhs) && mask_ != "m0_") {
                line(s.lhs + " = blend(" + s.lhs + ", " + vexpr(s) + ", " + mask_ + ");");
            } else {
                line(s.lhs + " = " + vexpr(s) + ";");
            }
            break;
        }
        case Stmt::Store: {
            const std::string& p = s.lhs;
            const std::string& v = s.e.args[0];
            const Ty pt = atomTy(p);
            const Ty et = elemOf(pt);
            if (auto it = aff_.find(p); it != aff_.end()) {
                const std::string val = vec(v, scalarKind(et));
                line(it->second == 1 ? "storec(" + p + ", " + val + ", " + mask_ + ");"
                                     : "store(affinePtr(" + p + ", " + std::to_string(it->second) + "), " + val +
                                           ", " + mask_ + ");");
            } else if (varying_.count(p)) {
                line("store(" + p + ", " + vec(v, scalarKind(et)) + ", " + mask_ + ");");
            } else if (!isLiteral(v) && (varying_.count(v) || aff_.count(v))) {
                line("*(" + p + ") = static_cast<" + cElem(pt) + ">(lastActive(" + vec(v, scalarKind(et)) + ", " +
                     mask_ + "));");
            } else {
                line("*(" + p + ") = static_cast<" + cElem(pt) + ">(" + v + ");");
            }
            break;
        }
        case Stmt::If: {
            if (!varying_.count(s.cond)) {
                line("if (" + s.cond + ") {");
                ++indent_;
                emitBlock(s.a);
                --indent_;
                if (!s.b.empty()) {
                    line("} else {");
                    ++indent_;
                    emitBlock(s.b);
                    --indent_;
                }
                line("}");
                break;
            }
            const std::string outer = mask_;
            const std::string mt = "m" + std::to_string(++mask_id_) + "_";
            const std::string me = "m" + std::to_string(++mask_id_) + "_";
            line("{");
            ++indent_;
            line("const VI " + mt + " = mAnd(" + outer + ", " + s.cond + ");");
            if (!s.b.empty()) line("const VI " + me + " = mAndNot(" + s.cond + ", " + outer + ");");
            line("if (any(" + mt + ")) {");
            ++indent_;
            mask_ = mt;
            emitBlock(s.a);
            mask_ = outer;
            --indent_;
            line("}");
            if (!s.b.empty()) {
                line("if (any(" + me + ")) {");
                ++indent_;
                mask_ = me;
                emitBlock(s.b);
                mask_ = outer;
                --indent_;
                line("}");
            }
            --indent_;
            line("}");
            break;
        }
        case Stmt::Loop: {
            ++loop_depth_;
            struct LoopDepth {
                int& d;
                ~LoopDepth() { --d; }
            } depth_guard{loop_depth_};
            if (!varying_.count(s.cond)) {
                line("while (true) {");
                ++indent_;
                emitBlock(s.a);
                line("if (!(" + s.cond + ")) break;");
                emitBlock(s.b);
                --indent_;
                line("}");
                break;
            }
            const std::string outer = mask_;
            const std::string ml = "m" + std::to_string(++mask_id_) + "_";
            line("{");
            ++indent_;
            line("VI " + ml + " = " + outer + ";");
            line("while (true) {");
            ++indent_;
            mask_ = ml;
            emitBlock(s.a);
            line(ml + " = mAnd(" + ml + ", " + s.cond + ");");
            line("if (!any(" + ml + ")) break;");
            emitBlock(s.b);
            mask_ = outer;
            --indent_;
            line("}");
            --indent_;
            line("}");
            break;
        }
        }
    }

    std::string declare(const std::string& v, Ty t) const {
        if (!varying_.count(v)) {
            switch (t) {
            case Ty::Int: return "int " + v + " = 0;";
            case Ty::Float: return "float " + v + " = 0;";
            case Ty::Bool: return "bool " + v + " = false;";
            case Ty::Char: return "char " + v + " = 0;";
            default: return std::string(cElem(t)) + "* " + v + " = nullptr;";
            }
        }
        switch (t) {
        case Ty::Float: return "VF " + v + " = zeroF();";
        case Ty::Int:
        case Ty::Bool:
        case Ty::Char: return "VI " + v + " = zeroI();";
        default: return "VPtr<" + std::string(cElem(t)) + "> " + v + "{};";
        }
    }

    std::string emit() {
        std::string params, call, unpack;
        int gi = 0, xi = 0;
        for (const Param& p : k_.params) {
            if (p.builtin) continue;
            const std::string ty = isPtr(p.ty) ? std::string(cElem(p.ty)) + "*" : p.ty == Ty::Int ? "int" : "float";
            params += ty + " " + p.name + ", ";
            call += p.name + ", ";
            if (isPtr(p.ty))
                unpack += "    " + ty + " const " + p.name + " = reinterpret_cast<" + ty + ">(a->gpr[" +
                          std::to_string(gi++) + "]);\n";
            else if (p.ty == Ty::Int)
                unpack += "    const int " + p.name + " = static_cast<int>(static_cast<int64_t>(a->gpr[" +
                          std::to_string(gi++) + "]));\n";
            else
                unpack += "    const float " + p.name + " = std::bit_cast<float>(static_cast<uint32_t>(a->xmm[" +
                          std::to_string(xi++) + "]));\n";
        }
        std::string body;
        indent_ = 1;
        mask_ = "m0_";
        out_.clear();
        line("using namespace kc::simd;");
        for (const auto& [v, t] : k_.vars) line(declare(v, t));
        line("const VI m0_ = tailMask(t0_, n_);");
        emitBlock(top_);
        body = out_;

        std::string s = "// Generated by kc_mslgen from Booth's --metal output for '" + name_ +
                        "'. Do not edit.\n// 8 CUDA threads per AVX2 register. Uniform values stay scalar.\n"
                        "#include <kernelcore/kernel_registry.hpp>\n#include <bit>\n#include <cstdint>\n"
                        "#include <metal_stdlib>\n#include \"kc_simd.hpp\"\n\n"
                        "namespace {\n#pragma warning(push)\n#pragma warning(disable : 4100 4189 4244 4245 4305 4700)\n"
                        "template <bool kVnni_>  // AVX-VNNI available: chosen once per launch\n"
                        "void group(" + params + "unsigned t0_, unsigned n_) {\n" + body +
                        "}\n#pragma warning(pop)\n} // namespace\n\n"
                        "extern \"C\" void kc_ksimd_" + name_ + "(const kc::LaunchArgs* a) {\n" + unpack +
                        "    const unsigned n = static_cast<unsigned>(a->nthreads);\n";
        if (body.find("vdp4aT<") != std::string::npos)
            s += "    if (kc::simd::cpuHasAvxVnni()) {\n"
                 "        for (unsigned t0 = 0; t0 < n; t0 += kc::simd::kLanes) group<true>(" + call + "t0, n);\n"
                 "        return;\n    }\n";
        s += "    for (unsigned t0 = 0; t0 < n; t0 += kc::simd::kLanes) group<false>(" + call + "t0, n);\n}\n";
        return s;
    }

    const Kernel&                      k_;
    const std::map<int, IrInst>&       ir_;
    std::string                        name_;
    std::map<std::string, Ty>          types_;
    std::vector<Region>                regions_;
    std::vector<Stmt>                  top_;
    std::set<std::string>              varying_{"tid"};  // Gen: held as vectors
    std::map<std::string, Shape>       shape_;
    std::map<std::string, int>         aff_;  // Aff: scalar lane-0 value, constant stride
    std::set<std::string>              blend_;
    std::map<std::string, std::string> ptr_base_;
    std::string                        out_, mask_;
    int                                indent_ = 0, mask_id_ = 0, loop_depth_ = 0;
};

int run(int argc, char** argv) {
    std::string metal, irp, name, native, simd;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--metal") metal = argv[i + 1];
        else if (a == "--ir") irp = argv[i + 1];
        else if (a == "--kernel") name = argv[i + 1];
        else if (a == "--native") native = argv[i + 1];
        else if (a == "--simd") simd = argv[i + 1];
        else die("unknown option " + a);
    }
    if (metal.empty() || irp.empty() || name.empty() || native.empty() || simd.empty())
        die("usage: kc_mslgen --metal F --ir F --kernel NAME --native OUT.cpp --simd OUT.cpp");

    std::vector<std::string> lines = splitLines(readFile(metal));
    for (std::string& l : lines) l = signedLiterals(l);
    const auto ir = parseIr(readFile(irp), name);
    const Kernel k = parseMetal(lines, name);
    std::string simd_block;
    const std::string fixed = fixMetal(lines, k, ir, simd_block);

    std::string simd_text, note = simd_block;
    if (note.empty()) {
        try {
            simd_text = Translator(k, ir, name).run();
        } catch (const Unsupported& u) {
            note = u.what();
        }
    }
    const bool has_simd = note.empty();
    if (!has_simd)
        simd_text = "// kc_mslgen: no simd path for '" + name + "': " + note + "\n// The native path is used.\n";
    writeFile(native, nativeFile(name, fixed, k, has_simd, note));
    writeFile(simd, simd_text);
    std::printf("kc_mslgen: %s: native ok; simd %s\n", name.c_str(),
                has_simd ? "ok" : ("unavailable (" + note + ")").c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kc_mslgen: error: %s\n", e.what());
        return 1;
    }
}
