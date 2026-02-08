#include "mla_kernel.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void validate_tensor(const torch::Tensor& t, const std::string& name, torch::ScalarType dtype, int dims) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dtype, name, " must have dtype ", dtype);
    TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

py::dict run(
    torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
    torch::Tensor kpe_cache, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    float sm_scale
) {
    validate_tensor(q_nope, "q_nope", torch::kBFloat16, 3);
    validate_tensor(q_pe, "q_pe", torch::kBFloat16, 3);
    validate_tensor(ckv_cache, "ckv_cache", torch::kBFloat16, 3);
    validate_tensor(kpe_cache, "kpe_cache", torch::kBFloat16, 3);
    validate_tensor(kv_indptr, "kv_indptr", torch::kInt32, 1);
    validate_tensor(kv_indices, "kv_indices", torch::kInt32, 1);

    const int batch_size = q_nope.size(0);
    auto output = torch::empty_like(q_nope);
    auto lse = torch::empty({batch_size, 16}, q_nope.options().dtype(torch::kFloat32));

    MlaPagedDecodeParams params;
    params.q_nope_ptr = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr());
    params.q_pe_ptr = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr());
    params.ckv_cache_ptr = reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr());
    params.kpe_cache_ptr = reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr());
    params.kv_indptr_ptr = kv_indptr.data_ptr<int>();
    params.kv_indices_ptr = kv_indices.data_ptr<int>();
    params.sm_scale = sm_scale;
    params.output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    params.lse_ptr = lse.data_ptr<float>();
    params.batch_size = batch_size;

    mla_paged_decode_launch(params, at::cuda::getCurrentCUDAStream());
    
    py::dict result;
    result["output"] = output;
    result["lse"] = lse;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MLA Paged Decode Kernel",
          py::arg("q_nope"), py::arg("q_pe"), py::arg("ckv_cache"),
          py::arg("kpe_cache"), py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"));
}