// Minimal .npy reader/writer: C-order arrays, little-endian numeric dtypes, versions
// 1.0 to 3.0. Enough to move tensors between the Python orchestrator and the harness
// without a JSON or numpy dependency on the C++ side.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ak_kernel.h"

namespace npy {

struct Array {
    int dtype = AK_F32;
    size_t itemsize = 4;
    std::vector<int64_t> shape;
    std::vector<char> bytes;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        return n;
    }
};

inline const char* descr_for(int dtype, size_t* itemsize) {
    switch (dtype) {
        case AK_F32: *itemsize = 4; return "<f4";
        case AK_F64: *itemsize = 8; return "<f8";
        case AK_F16: *itemsize = 2; return "<f2";
        case AK_BF16: *itemsize = 2; return "<u2";  // numpy has no bf16; raw bits
        case AK_I32: *itemsize = 4; return "<i4";
        case AK_I64: *itemsize = 8; return "<i8";
        case AK_U8: *itemsize = 1; return "|u1";
        case AK_I8: *itemsize = 1; return "|i1";
        case AK_BOOL: *itemsize = 1; return "|b1";
    }
    throw std::runtime_error("npy: unknown ak_dtype " + std::to_string(dtype));
}

inline int dtype_for(const std::string& descr, size_t* itemsize) {
    std::string d = descr;
    if (!d.empty() && (d[0] == '<' || d[0] == '=' || d[0] == '|')) d = d.substr(1);
    else if (!d.empty() && d[0] == '>') throw std::runtime_error("npy: big-endian data unsupported");
    if (d == "f4") { *itemsize = 4; return AK_F32; }
    if (d == "f8") { *itemsize = 8; return AK_F64; }
    if (d == "f2") { *itemsize = 2; return AK_F16; }
    if (d == "i4") { *itemsize = 4; return AK_I32; }
    if (d == "i8") { *itemsize = 8; return AK_I64; }
    if (d == "u1") { *itemsize = 1; return AK_U8; }
    if (d == "i1") { *itemsize = 1; return AK_I8; }
    if (d == "b1") { *itemsize = 1; return AK_BOOL; }
    if (d == "u2") { *itemsize = 2; return AK_BF16; }  // raw bits, see descr_for
    throw std::runtime_error("npy: unsupported dtype descr '" + descr + "'");
}

inline std::string header_value(const std::string& header, const std::string& key) {
    size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npy: header missing " + key);
    size_t colon = header.find(':', k);
    size_t start = header.find_first_not_of(" ", colon + 1);
    if (header[start] == '\'') {
        size_t end = header.find('\'', start + 1);
        return header.substr(start + 1, end - start - 1);
    }
    if (header[start] == '(') {
        size_t end = header.find(')', start);
        return header.substr(start, end - start + 1);
    }
    size_t end = header.find_first_of(",}", start);
    return header.substr(start, end - start);
}

inline Array load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic in " + path);
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (uint32_t(b[3]) << 24);
    }
    std::string header(hlen, '\0');
    f.read(&header[0], hlen);

    Array a;
    a.dtype = dtype_for(header_value(header, "descr"), &a.itemsize);
    if (header_value(header, "fortran_order") != "False")
        throw std::runtime_error("npy: fortran_order arrays unsupported: " + path);
    std::string shape = header_value(header, "shape");  // "(2048, 1024)" or "()" or "(5,)"
    size_t pos = 1;
    while (pos < shape.size()) {
        size_t next = shape.find_first_of(",)", pos);
        std::string tok = shape.substr(pos, next - pos);
        size_t ns = tok.find_first_not_of(" ");
        if (ns != std::string::npos) a.shape.push_back(std::stoll(tok.substr(ns)));
        if (next == std::string::npos || shape[next] == ')') break;
        pos = next + 1;
    }
    a.bytes.resize(size_t(a.numel()) * a.itemsize);
    f.read(a.bytes.data(), std::streamsize(a.bytes.size()));
    if (size_t(f.gcount()) != a.bytes.size()) throw std::runtime_error("npy: truncated data in " + path);
    return a;
}

inline void save(const std::string& path, const Array& a) {
    size_t itemsize = 0;
    std::string header = "{'descr': '" + std::string(descr_for(a.dtype, &itemsize)) +
                         "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < a.shape.size(); ++i) {
        header += std::to_string(a.shape[i]);
        if (a.shape.size() == 1 || i + 1 < a.shape.size()) header += ",";
        if (i + 1 < a.shape.size()) header += " ";
    }
    header += "), }";
    // Total header (magic + version + len + dict + '\n') padded to a multiple of 64.
    size_t prefix = 6 + 2 + 2;
    size_t pad = 64 - ((prefix + header.size() + 1) % 64);
    if (pad == 64) pad = 0;
    header += std::string(pad, ' ') + "\n";
    if (header.size() > 65535) throw std::runtime_error("npy: header too long");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot write " + path);
    f.write("\x93NUMPY\x01\x00", 8);
    unsigned char len[2] = {static_cast<unsigned char>(header.size() & 0xff),
                            static_cast<unsigned char>(header.size() >> 8)};
    f.write(reinterpret_cast<char*>(len), 2);
    f.write(header.data(), std::streamsize(header.size()));
    f.write(a.bytes.data(), std::streamsize(a.bytes.size()));
    if (!f) throw std::runtime_error("npy: short write to " + path);
}

}  // namespace npy
