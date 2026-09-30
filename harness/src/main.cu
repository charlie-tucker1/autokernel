// Evaluator process for the cuda_cpp backend. See docs/ARCHITECTURE.md, "Backends" and
// "Evaluation protocol".
//
//   harness describe
//       Print the device's properties as JSON.
//   harness eval --lib cand.so [--entry ak_kernel] --result out.json
//       [--warmup N] [--trials N] [--flush-l2 0|1]
//       --check-in NAME=a.npy ... --check-out NAME=c.npy ...
//       --time-in  NAME=a2.npy ... --time-out NAME=c2.npy ...
//       Load the candidate, run it once on the check tensors (outputs written back to
//       the --check-out files), then warm up and time it on the time tensors (outputs
//       written back to the --time-out files). Timings and status go to --result.
//
// The candidate never sees this process's data beyond the tensors it is handed, and
// this process trusts nothing the candidate reports except its return code.
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ak_kernel.h"
#include "npy.h"

namespace {

// ---------------------------------------------------------------- JSON helpers

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += char(c);
                }
        }
    }
    return o + "\"";
}

// ---------------------------------------------------------------- device info

struct DeviceInfo {
    std::string json;
    int l2_bytes = 0;
};

int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, a, dev);
    return v;
}

DeviceInfo describe_device(int dev) {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    char pci[32] = {0};
    cudaDeviceGetPCIBusId(pci, sizeof pci, dev);
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    const int sm_clock_khz = attr(cudaDevAttrClockRate, dev);
    const int mem_clock_khz = attr(cudaDevAttrMemoryClockRate, dev);
    const int bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
    const double nominal_bw = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    DeviceInfo d;
    d.l2_bytes = attr(cudaDevAttrL2CacheSize, dev);
    char buf[2048];
    std::snprintf(buf, sizeof buf,
        "{\"name\": %s, \"index\": %d, \"pci_bus_id\": %s, "
        "\"cc_major\": %d, \"cc_minor\": %d, \"arch\": \"sm_%d%d\", "
        "\"sm_count\": %d, \"l2_bytes\": %d, "
        "\"smem_per_block\": %d, \"smem_per_block_optin\": %d, \"smem_per_sm\": %d, "
        "\"regs_per_block\": %d, \"regs_per_sm\": %d, \"warp_size\": %d, "
        "\"max_threads_per_block\": %d, \"max_threads_per_sm\": %d, \"max_blocks_per_sm\": %d, "
        "\"sm_clock_khz\": %d, \"mem_clock_khz\": %d, \"mem_bus_width_bits\": %d, "
        "\"nominal_mem_bw_gbs\": %.1f, \"total_mem_bytes\": %zu, "
        "\"driver_version\": %d, \"runtime_version\": %d}",
        jstr(p.name).c_str(), dev, jstr(pci).c_str(),
        p.major, p.minor, p.major, p.minor,
        p.multiProcessorCount, d.l2_bytes,
        attr(cudaDevAttrMaxSharedMemoryPerBlock, dev), attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
        attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
        attr(cudaDevAttrMaxRegistersPerBlock, dev), attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev),
        attr(cudaDevAttrWarpSize, dev),
        attr(cudaDevAttrMaxThreadsPerBlock, dev), attr(cudaDevAttrMaxThreadsPerMultiProcessor, dev),
        attr(cudaDevAttrMaxBlocksPerMultiprocessor, dev),
        sm_clock_khz, mem_clock_khz, bus_bits, nominal_bw, total_b, drv, rt);
    d.json = buf;
    return d;
}

// ---------------------------------------------------------------- options

struct TensorFile {
    std::string name, path;
};

struct Options {
    std::string lib, entry = "ak_kernel", result;
    int warmup = 3, trials = 20, device = 0;
    bool flush_l2 = true;
    std::vector<TensorFile> check_in, check_out, time_in, time_out;
};

TensorFile parse_tensor_arg(const std::string& s) {
    size_t eq = s.find('=');
    if (eq == std::string::npos) throw std::runtime_error("expected NAME=PATH, got " + s);
    return {s.substr(0, eq), s.substr(eq + 1)};
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--lib") o.lib = next();
        else if (a == "--entry") o.entry = next();
        else if (a == "--result") o.result = next();
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--trials") o.trials = std::stoi(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--flush-l2") o.flush_l2 = std::stoi(next()) != 0;
        else if (a == "--check-in") o.check_in.push_back(parse_tensor_arg(next()));
        else if (a == "--check-out") o.check_out.push_back(parse_tensor_arg(next()));
        else if (a == "--time-in") o.time_in.push_back(parse_tensor_arg(next()));
        else if (a == "--time-out") o.time_out.push_back(parse_tensor_arg(next()));
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.lib.empty() || o.result.empty()) throw std::runtime_error("--lib and --result are required");
    if (o.check_out.empty()) throw std::runtime_error("at least one --check-out is required");
    if (o.time_in.size() != o.check_in.size() || o.time_out.size() != o.check_out.size())
        throw std::runtime_error("--time-in/--time-out must mirror --check-in/--check-out");
    if (o.trials < 1) throw std::runtime_error("--trials must be >= 1");
    return o;
}

// ---------------------------------------------------------------- result

struct Result {
    std::string status = "ok", stage = "start", error;
    int rc = 0;
    float check_launch_ms = 0;
    std::vector<float> trial_ms;
    std::string device_json = "null";
    size_t flush_bytes = 0;
    int abi_version = -1;
    double wall_ms = 0;
    const Options* opts = nullptr;

    void write() const {
        if (!opts || opts->result.empty()) return;
        FILE* f = std::fopen(opts->result.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "{\"status\": %s, \"stage\": %s, \"error\": %s, \"rc\": %d, ",
                     jstr(status).c_str(), jstr(stage).c_str(), jstr(error).c_str(), rc);
        std::fprintf(f, "\"abi_version\": %d, \"warmup\": %d, \"trials\": %d, \"flush_l2\": %s, "
                     "\"flush_bytes\": %zu, \"check_launch_ms\": %.6f, \"wall_ms\": %.3f, ",
                     abi_version, opts->warmup, opts->trials, opts->flush_l2 ? "true" : "false",
                     flush_bytes, check_launch_ms, wall_ms);
        std::fprintf(f, "\"trial_ms\": [");
        for (size_t i = 0; i < trial_ms.size(); ++i)
            std::fprintf(f, "%s%.6f", i ? ", " : "", trial_ms[i]);
        std::fprintf(f, "], \"device\": %s}\n", device_json.c_str());
        std::fclose(f);
    }
};

Result g_result;
std::chrono::steady_clock::time_point g_t0;

[[noreturn]] void fail(const std::string& stage, const std::string& msg, int exit_code) {
    g_result.status = "error";
    g_result.stage = stage;
    g_result.error = msg;
    g_result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
    g_result.write();
    std::fprintf(stderr, "harness: %s: %s\n", stage.c_str(), msg.c_str());
    std::exit(exit_code);
}

void check(cudaError_t e, const std::string& stage, const char* what) {
    if (e != cudaSuccess)
        fail(stage, std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")", 3);
}

// ---------------------------------------------------------------- tensors

struct DeviceTensor {
    ak_tensor t{};
    npy::Array host;
    std::string name, path;
    size_t bytes = 0;
};

DeviceTensor upload(const TensorFile& tf, const std::string& stage) {
    DeviceTensor d;
    d.name = tf.name;
    d.path = tf.path;
    try {
        d.host = npy::load(tf.path);
    } catch (const std::exception& e) {
        fail(stage, e.what(), 4);
    }
    if (d.host.shape.size() > AK_MAX_DIMS) fail(stage, "tensor " + tf.name + " has too many dims", 4);
    d.bytes = d.host.bytes.size();
    d.t.dtype = d.host.dtype;
    d.t.ndim = int32_t(d.host.shape.size());
    int64_t stride = 1;
    for (int i = d.t.ndim - 1; i >= 0; --i) {
        d.t.shape[i] = d.host.shape[i];
        d.t.strides[i] = stride;
        stride *= d.host.shape[i];
    }
    check(cudaMalloc(&d.t.data, d.bytes ? d.bytes : 1), stage, "cudaMalloc");
    check(cudaMemcpy(d.t.data, d.host.bytes.data(), d.bytes, cudaMemcpyHostToDevice), stage, "cudaMemcpy H2D");
    return d;
}

void download_and_save(DeviceTensor& d, const std::string& stage) {
    check(cudaMemcpy(d.host.bytes.data(), d.t.data, d.bytes, cudaMemcpyDeviceToHost), stage, "cudaMemcpy D2H");
    try {
        npy::save(d.path, d.host);
    } catch (const std::exception& e) {
        fail(stage, e.what(), 4);
    }
}

void poison(std::vector<DeviceTensor>& outs, cudaStream_t s, const std::string& stage) {
    for (auto& o : outs) check(cudaMemsetAsync(o.t.data, 0xFF, o.bytes, s), stage, "poison outputs");
}

std::vector<ak_tensor> views(const std::vector<DeviceTensor>& ts) {
    std::vector<ak_tensor> v;
    for (const auto& t : ts) v.push_back(t.t);
    return v;
}

// ---------------------------------------------------------------- main

using entry_fn = int (*)(const ak_tensor*, int32_t, ak_tensor*, int32_t, cudaStream_t);
using abi_fn = int (*)(void);

int run_eval(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "harness eval: %s\n", e.what());
        return 1;
    }
    g_result.opts = &o;

    check(cudaSetDevice(o.device), "init", "cudaSetDevice");
    DeviceInfo dev = describe_device(o.device);
    g_result.device_json = dev.json;

    // Load the candidate.
    void* h = dlopen(o.lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) fail("load", dlerror(), 2);
    auto entry = reinterpret_cast<entry_fn>(dlsym(h, o.entry.c_str()));
    if (!entry) fail("load", "symbol " + o.entry + " not found: " + dlerror(), 2);
    if (auto abi = reinterpret_cast<abi_fn>(dlsym(h, "ak_abi_version"))) {
        g_result.abi_version = abi();
        if (g_result.abi_version != AK_ABI_VERSION)
            fail("load", "candidate ABI version " + std::to_string(g_result.abi_version) +
                         " != harness " + std::to_string(AK_ABI_VERSION), 2);
    }

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "init", "cudaStreamCreate");
    cudaEvent_t ev_start, ev_stop;
    check(cudaEventCreate(&ev_start), "init", "cudaEventCreate");
    check(cudaEventCreate(&ev_stop), "init", "cudaEventCreate");

    auto launch = [&](std::vector<ak_tensor>& in, std::vector<ak_tensor>& out, const std::string& stage) {
        int rc = entry(in.data(), int32_t(in.size()), out.data(), int32_t(out.size()), stream);
        if (rc != 0) {
            g_result.rc = rc;
            fail(stage, "candidate returned " + std::to_string(rc), 3);
        }
        check(cudaGetLastError(), stage, "launch");
    };

    // ---- correctness launch: one call on the check tensors, outputs written back.
    {
        g_result.stage = "check";
        std::vector<DeviceTensor> ins, outs;
        for (const auto& tf : o.check_in) ins.push_back(upload(tf, "check"));
        for (const auto& tf : o.check_out) outs.push_back(upload(tf, "check"));
        auto iv = views(ins), ov = views(outs);
        poison(outs, stream, "check");
        check(cudaEventRecord(ev_start, stream), "check", "event");
        launch(iv, ov, "check");
        check(cudaEventRecord(ev_stop, stream), "check", "event");
        check(cudaStreamSynchronize(stream), "check", "sync after correctness launch");
        check(cudaEventElapsedTime(&g_result.check_launch_ms, ev_start, ev_stop), "check", "elapsed");
        for (auto& t : outs) download_and_save(t, "check");
        for (auto& t : ins) cudaFree(t.t.data);
        for (auto& t : outs) cudaFree(t.t.data);
    }

    // ---- timing: fresh tensors, warmup, then one launch per trial between events.
    {
        g_result.stage = "timing";
        std::vector<DeviceTensor> ins, outs;
        for (const auto& tf : o.time_in) ins.push_back(upload(tf, "timing"));
        for (const auto& tf : o.time_out) outs.push_back(upload(tf, "timing"));
        auto iv = views(ins), ov = views(outs);

        void* flush = nullptr;
        if (o.flush_l2) {
            g_result.flush_bytes = std::max<size_t>(size_t(dev.l2_bytes) * 2, size_t(32) << 20);
            check(cudaMalloc(&flush, g_result.flush_bytes), "timing", "cudaMalloc flush buffer");
        }

        for (int i = 0; i < o.warmup; ++i) {
            poison(outs, stream, "warmup");
            launch(iv, ov, "warmup");
        }
        check(cudaStreamSynchronize(stream), "warmup", "sync after warmup");

        g_result.trial_ms.reserve(o.trials);
        for (int i = 0; i < o.trials; ++i) {
            poison(outs, stream, "timing");
            if (flush) check(cudaMemsetAsync(flush, 0, g_result.flush_bytes, stream), "timing", "flush L2");
            check(cudaEventRecord(ev_start, stream), "timing", "event");
            launch(iv, ov, "timing");
            check(cudaEventRecord(ev_stop, stream), "timing", "event");
            check(cudaStreamSynchronize(stream), "timing", "sync after trial");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "timing", "elapsed");
            g_result.trial_ms.push_back(ms);
        }
        for (auto& t : outs) download_and_save(t, "timing");
        if (flush) cudaFree(flush);
        for (auto& t : ins) cudaFree(t.t.data);
        for (auto& t : outs) cudaFree(t.t.data);
    }

    g_result.stage = "done";
    g_result.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
    g_result.write();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    g_t0 = std::chrono::steady_clock::now();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "describe") {
        int dev = 0;
        for (int i = 2; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--device") dev = std::atoi(argv[i + 1]);
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || dev >= count) {
            std::fprintf(stderr, "harness: no usable CUDA device %d\n", dev);
            return 3;
        }
        std::printf("%s\n", describe_device(dev).json.c_str());
        return 0;
    }
    if (mode == "eval") return run_eval(argc, argv);
    std::fprintf(stderr, "usage: harness describe [--device N]\n"
                         "       harness eval --lib cand.so --result out.json [options] "
                         "--check-in N=P ... --check-out N=P ... --time-in N=P ... --time-out N=P ...\n");
    return 1;
}
