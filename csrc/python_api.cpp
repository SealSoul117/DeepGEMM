#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/runtime.hpp"
#include "apis/mega_moe_trace_api.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::runtime::register_apis(m);

    m.def("init_mega_moe_trace_buffer", &deep_gemm::init_mega_moe_trace_buffer,
      pybind11::arg("num_sms"),
      "Allocate the MegaMoE trace ringbuffer on the device. Call once at program start.");

    m.def("reset_mega_moe_trace", &deep_gemm::reset_mega_moe_trace,
        "Zero out the MegaMoE trace buffer (async, no host sync).");

    m.def("dump_mega_moe_trace", &deep_gemm::dump_mega_moe_trace,
        pybind11::arg("path"),
        "Synchronize, copy trace buffer to host, write to file. "
        "Must NOT be called inside a perf-measurement loop.");
}
