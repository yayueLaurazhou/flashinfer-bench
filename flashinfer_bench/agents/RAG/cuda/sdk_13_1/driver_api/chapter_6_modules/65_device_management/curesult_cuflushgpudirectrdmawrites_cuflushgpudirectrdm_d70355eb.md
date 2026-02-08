# CUresult cuFlushGPUDirectRDMAWrites (CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope)

Blocks until remote writes are visible to the specified scope.

###### Parameters

**target**

  - The target of the operation, see CUflushGPUDirectRDMAWritesTarget
**scope**

  - The scope of the operation, see CUflushGPUDirectRDMAWritesScope

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

###### Description

Blocks until GPUDirect RDMA writes to the target context via mappings created through APIs like
nvidia_p2p_get_pages (see [https://docs.nvidia.com/cuda/gpudirect-rdma](https://docs.nvidia.com/cuda/gpudirect-rdma) for more information), are
visible to the specified scope.

If the scope equals or lies within the scope indicated by
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, the call will be a no-op
and can be safely omitted for performance. This can be determined by comparing the numerical values
between the two enums, with smaller scopes having smaller values.

On platforms that support GPUDirect RDMA writes via more than one path in hardware (see
CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE), the user should consider
those paths as belonging to separate ordering domains. Note that in such cases CUDA driver


CUDA Driver API TRM-06703-001 _vRelease Version  |  111


Modules


will report both RDMA writes ordering and RDMA write scope as ALL_DEVICES and a call to
cuFlushGPUDirectRDMA will be a no-op, but when these multiple paths are used simultaneously, it is
the user's responsibility to ensure ordering by using mechanisms outside the scope of CUDA.

Users may query support for this API via
CU_DEVICE_ATTRIBUTE_FLUSH_FLUSH_GPU_DIRECT_RDMA_OPTIONS.


Note:


Note that this function may also return error codes from previous, asynchronous launches.