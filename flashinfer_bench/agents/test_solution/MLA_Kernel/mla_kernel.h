#pragma once

// #include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Struct to hold tensor pointers and metadata
struct MlaPagedDecodeParams {
    // Inputs
    const __nv_bfloat16* q_nope_ptr;
    const __nv_bfloat16* q_pe_ptr;
    const __nv_bfloat16* ckv_cache_ptr;
    const __nv_bfloat16* kpe_cache_ptr;
    const int* kv_indptr_ptr;
    const int* kv_indices_ptr;
    float sm_scale;

    // Outputs
    __nv_bfloat16* output_ptr;
    float* lse_ptr;

    // Dimensions
    int batch_size;
};

// Host function to launch the CUDA kernel
void mla_paged_decode_launch(const MlaPagedDecodeParams& params, cudaStream_t stream);