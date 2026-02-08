#include "mla_kernel.h"
#include <cmath>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// --- Constants based on specification ---
constexpr int kNumQoHeads = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kPageSize = 1;

// --- Kernel Tuning Parameters ---
constexpr int kBlockThreads = 256;
constexpr int kTileK = 16;
constexpr int kWarpSize = 32;

// --- Derived Constants ---
constexpr int kAccVecsPerThread = kHeadDimCkv / kBlockThreads;
constexpr int kNumWarps = kBlockThreads / kWarpSize;

// --- Device Helpers ---

__device__ __forceinline__ float warp_reduce_sum(float val, const cg::thread_block_tile<kWarpSize>& warp) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val, const cg::thread_block_tile<kWarpSize>& warp) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = max(val, warp.shfl_down(val, offset));
    }
    return val;
}

// --- Main Kernel Implementation ---

__global__ void __launch_bounds__(kBlockThreads)
mla_paged_decode_kernel(const MlaPagedDecodeParams params) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (batch_idx >= params.batch_size) return;

    extern __shared__ char smem[];
    __nv_bfloat16* q_c_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* q_p_smem = q_c_smem + kHeadDimCkv;
    __nv_bfloat16* k_c_tile_smem = q_p_smem + kHeadDimKpe;
    __nv_bfloat16* k_p_tile_smem = k_c_tile_smem + kTileK * kHeadDimCkv;
    
    float* float_smem_base = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(k_p_tile_smem + kTileK * kHeadDimKpe + 3) & ~3);
    float* logits_smem = float_smem_base;
    float* attn_smem = logits_smem + kTileK;
    float* scratch_smem = attn_smem + kTileK;

    const int thread_id = threadIdx.x;
    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<kWarpSize> warp = cg::tiled_partition<kWarpSize>(block);
    const int warp_id = thread_id / kWarpSize;
    const int lane_id = thread_id % kWarpSize;

    const __nv_bfloat16* q_c_gmem = params.q_nope_ptr + (batch_idx * kNumQoHeads + head_idx) * kHeadDimCkv;
    const __nv_bfloat16* q_p_gmem = params.q_pe_ptr + (batch_idx * kNumQoHeads + head_idx) * kHeadDimKpe;

    for (int i = thread_id; i < kHeadDimCkv / 2; i += kBlockThreads) {
        reinterpret_cast<__nv_bfloat162*>(q_c_smem)[i] = reinterpret_cast<const __nv_bfloat162*>(q_c_gmem)[i];
    }
    for (int i = thread_id; i < kHeadDimKpe / 2; i += kBlockThreads) {
        reinterpret_cast<__nv_bfloat162*>(q_p_smem)[i] = reinterpret_cast<const __nv_bfloat162*>(q_p_gmem)[i];
    }

    const int page_start_offset = params.kv_indptr_ptr[batch_idx];
    const int page_end_offset = params.kv_indptr_ptr[batch_idx + 1];
    const int seq_len = page_end_offset - page_start_offset;

    if (seq_len <= 0) {
        __nv_bfloat16* out_ptr = params.output_ptr + (batch_idx * kNumQoHeads + head_idx) * kHeadDimCkv;
        for (int i = thread_id; i < kHeadDimCkv / 2; i += kBlockThreads) {
            reinterpret_cast<__nv_bfloat162*>(out_ptr)[i] = __float2bfloat162_rn(0.0f);
        }
        if (thread_id == 0) params.lse_ptr[batch_idx * kNumQoHeads + head_idx] = -INFINITY;
        return;
    }
    block.sync();

    float o_acc[kAccVecsPerThread];
    for (int i = 0; i < kAccVecsPerThread; ++i) o_acc[i] = 0.0f;

    float max_logit = -INFINITY;
    float sum_exp = 0.0f;

    for (int tile_offset = 0; tile_offset < seq_len; tile_offset += kTileK) {
        const int current_tile_size = min(kTileK, seq_len - tile_offset);

        for (int i = thread_id; i < current_tile_size * (kHeadDimCkv / 2); i += kBlockThreads) {
            int k = i / (kHeadDimCkv / 2);
            int d_idx = i % (kHeadDimCkv / 2);
            int page_idx = params.kv_indices_ptr[page_start_offset + tile_offset + k];
            reinterpret_cast<__nv_bfloat162*>(k_c_tile_smem)[k * (kHeadDimCkv / 2) + d_idx] =
                reinterpret_cast<const __nv_bfloat162*>(params.ckv_cache_ptr + page_idx * kHeadDimCkv)[d_idx];
        }
        for (int i = thread_id; i < current_tile_size * (kHeadDimKpe / 2); i += kBlockThreads) {
            int k = i / (kHeadDimKpe / 2);
            int d_idx = i % (kHeadDimKpe / 2);
            int page_idx = params.kv_indices_ptr[page_start_offset + tile_offset + k];
            reinterpret_cast<__nv_bfloat162*>(k_p_tile_smem)[k * (kHeadDimKpe / 2) + d_idx] =
                reinterpret_cast<const __nv_bfloat162*>(params.kpe_cache_ptr + page_idx * kHeadDimKpe)[d_idx];
        }
        block.sync();

        for (int k_outer = 0; k_outer < current_tile_size; k_outer += kNumWarps) {
            const int k = k_outer + warp_id;
            if (k < current_tile_size) {
                float partial_sum = 0.0f;
                #pragma unroll
                for (int d = lane_id; d < kHeadDimCkv; d += kWarpSize) {
                    partial_sum += __bfloat162float(q_c_smem[d]) * __bfloat162float(k_c_tile_smem[k * kHeadDimCkv + d]);
                }
                #pragma unroll
                for (int d = lane_id; d < kHeadDimKpe; d += kWarpSize) {
                    partial_sum += __bfloat162float(q_p_smem[d]) * __bfloat162float(k_p_tile_smem[k * kHeadDimKpe + d]);
                }
                float total_logit = warp_reduce_sum(partial_sum, warp);
                if (lane_id == 0) logits_smem[k] = total_logit * params.sm_scale;
            }
        }
        block.sync();

        float tile_max_logit = -INFINITY;
        if (warp_id == 0) {
            float local_max = (lane_id < current_tile_size) ? logits_smem[lane_id] : -INFINITY;
            tile_max_logit = warp_reduce_max(local_max, warp);
            if (lane_id == 0) scratch_smem[0] = tile_max_logit;
        }
        block.sync();
        tile_max_logit = scratch_smem[0];

        float old_max_logit = max_logit;
        max_logit = max(max_logit, tile_max_logit);
        float scale = expf(old_max_logit - max_logit);
        sum_exp *= scale;

        if (warp_id == 0) {
            float local_sum = 0.0f;
            if (lane_id < current_tile_size) {
                float val = expf(logits_smem[lane_id] - max_logit);
                attn_smem[lane_id] = val;
                local_sum = val;
            }
            float tile_sum_exp = warp_reduce_sum(local_sum, warp);
            if (lane_id == 0) scratch_smem[0] = tile_sum_exp;
        }
        block.sync();
        sum_exp += scratch_smem[0];

        for (int i = 0; i < kAccVecsPerThread; ++i) o_acc[i] *= scale;
        for (int k = 0; k < current_tile_size; ++k) {
            float attn_k = attn_smem[k];
            for (int i = 0; i < kAccVecsPerThread; ++i) {
                int d = thread_id + i * kBlockThreads;
                if (d < kHeadDimCkv) {
                    o_acc[i] += attn_k * __bfloat162float(k_c_tile_smem[k * kHeadDimCkv + d]);
                }
            }
        }
        block.sync();
    }

    float inv_sum_exp = (sum_exp > 1e-8f) ? 1.0f / sum_exp : 0.0f;
    __nv_bfloat16* out_ptr = params.output_ptr + (batch_idx * kNumQoHeads + head_idx) * kHeadDimCkv;
    for (int i = 0; i < kAccVecsPerThread; ++i) {
        int d = thread_id + i * kBlockThreads;
        if (d < kHeadDimCkv) {
            out_ptr[d] = __float2bfloat16(o_acc[i] * inv_sum_exp);
        }
    }

    if (thread_id == 0) {
        const float log2_e = 1.44269504089f;
        float lse_val = (sum_exp > 1e-8f) ? max_logit * log2_e + log2f(sum_exp) : -INFINITY;
        params.lse_ptr[batch_idx * kNumQoHeads + head_idx] = lse_val;
    }
}

void mla_paged_decode_launch(const MlaPagedDecodeParams& params, cudaStream_t stream) {
    if (params.batch_size == 0) return;
    dim3 grid_dim(params.batch_size, kNumQoHeads);
    dim3 block_dim(kBlockThreads);
    size_t smem_size = (kHeadDimCkv + kHeadDimKpe) * sizeof(__nv_bfloat16) +
                       kTileK * (kHeadDimCkv + kHeadDimKpe) * sizeof(__nv_bfloat16) +
                       (kTileK * 2 + 16) * sizeof(float);
    mla_paged_decode_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params);
}