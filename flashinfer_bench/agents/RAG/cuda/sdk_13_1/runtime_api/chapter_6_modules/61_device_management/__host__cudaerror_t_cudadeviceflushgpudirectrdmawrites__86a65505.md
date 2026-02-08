# __host__cudaError_t cudaDeviceFlushGPUDirectRDMAWrites (cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope)

Blocks until remote writes are visible to the specified scope.

##### Parameters

**target**

  - The target of the operation, see cudaFlushGPUDirectRDMAWritesTarget
**scope**

  - The scope of the operation, see cudaFlushGPUDirectRDMAWritesScope

##### Returns

cudaSuccess, cudaErrorNotSupported,

##### Description

Blocks until remote writes to the target context via mappings created through GPUDirect RDMA APIs,
like nvidia_p2p_get_pages (see [https://docs.nvidia.com/cuda/gpudirect-rdma](https://docs.nvidia.com/cuda/gpudirect-rdma) for more information), are
visible to the specified scope.

If the scope equals or lies within the scope indicated by
cudaDevAttrGPUDirectRDMAWritesOrdering, the call will be a no-op and can be safely omitted for
performance. This can be determined by comparing the numerical values between the two enums, with
smaller scopes having smaller values.

Users may query support for this API via cudaDevAttrGPUDirectRDMAFlushWritesOptions.





CUDA Runtime API vRelease Version  |  11


Modules


See also:

cuFlushGPUDirectRDMAWrites