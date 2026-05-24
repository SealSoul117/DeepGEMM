#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <filesystem>

#include <deep_gemm/mega_moe_trace.cuh>

namespace deep_gemm {

// Process-global singleton state. One trace buffer per process.
struct MegaMoeTraceState {
    uint8_t* device_buf = nullptr;
    size_t   bytes      = 0;
    uint32_t num_sms    = 0;
    bool     initialized = false;
};

inline MegaMoeTraceState& mega_moe_trace_state() {
    static MegaMoeTraceState s;
    return s;
}

// Allocate the trace buffer once. Safe to call multiple times — second+ calls are no-ops.
inline void init_mega_moe_trace_buffer(uint32_t num_sms) {
    auto& s = mega_moe_trace_state();
    if (s.initialized) return;
    s.num_sms = num_sms;
    s.bytes   = num_sms * deep_gemm::trace::kBytesPerSm;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&s.device_buf), s.bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("init_mega_moe_trace_buffer cudaMalloc failed: ")
                                 + cudaGetErrorString(err));
    }
    cudaMemset(s.device_buf, 0, s.bytes);
    s.initialized = true;
}

// Zero out the counter+events region. Use this before a "clean" launch you want to trace.
// Uses MemsetAsync on default stream so it doesn't force a host sync (still gets ordered
// with subsequent launches in default stream, which is what we want).
inline void reset_mega_moe_trace() {
    auto& s = mega_moe_trace_state();
    if (not s.initialized) return;
    cudaMemsetAsync(s.device_buf, 0, s.bytes);
}

// Synchronize, copy device buffer to host, write to disk.
// Call this AFTER bench_kineto finishes — never inside its loop.
inline void dump_mega_moe_trace(const std::string& path) {
    auto& s = mega_moe_trace_state();
    if (not s.initialized) {
        throw std::runtime_error("dump_mega_moe_trace called before init_mega_moe_trace_buffer");
    }

        // Ensure parent directory exists
    auto parent = std::filesystem::path(path).parent_path();
    if (not parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw std::runtime_error("dump_mega_moe_trace: cannot create directory "
                                     + parent.string() + ": " + ec.message());
        }
    }
    
    cudaDeviceSynchronize();  // ensure all enqueued launches are done
    std::vector<uint8_t> raw(s.bytes);
    cudaError_t err = cudaMemcpy(raw.data(), s.device_buf, s.bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("dump_mega_moe_trace cudaMemcpy failed: ")
                                 + cudaGetErrorString(err));
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("dump_mega_moe_trace: cannot open " + path);
    }
    std::fwrite(&s.num_sms, sizeof(uint32_t), 1, f);
    std::fwrite(raw.data(), 1, s.bytes, f);
    std::fclose(f);
}

// Helper for the launch path: build a TraceBuf descriptor from the singleton.
// Returns a zero descriptor if not yet initialized (means: tracing disabled this run).
inline deep_gemm::trace::TraceBuf get_current_trace_buf_descriptor() {
    auto& s = mega_moe_trace_state();
    deep_gemm::trace::TraceBuf tb;
    tb.data = s.device_buf;          // nullptr if not initialized — kernel must tolerate that
    tb.num_sms = s.num_sms;
    return tb;
}

}  // namespace deep_gemm