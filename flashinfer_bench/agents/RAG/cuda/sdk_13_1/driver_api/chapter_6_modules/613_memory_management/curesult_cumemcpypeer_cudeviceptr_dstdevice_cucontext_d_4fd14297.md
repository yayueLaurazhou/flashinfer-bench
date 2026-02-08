# CUresult cuMemcpyPeer (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)

Copies device memory between two contexts.

###### Parameters

**dstDevice**

  - Destination device pointer
**dstContext**

  - Destination context
**srcDevice**

  - Source device pointer
**srcContext**

  - Source context
**ByteCount**

  - Size of memory copy in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Copies from device memory in one context to device memory in another context. dstDevice is
the base device pointer of the destination memory and dstContext is the destination context.
srcDevice is the base device pointer of the source memory and srcContext is the source pointer.
ByteCount specifies the number of bytes to copy.





See also:

cuMemcpyDtoD, cuMemcpy3DPeer, cuMemcpyDtoDAsync, cuMemcpyPeerAsync,
cuMemcpy3DPeerAsync, cudaMemcpyPeer


CUDA Driver API TRM-06703-001 _vRelease Version  |  238


Modules