# CUresult cuMemcpy3DPeer (const CUDA_MEMCPY3D_PEER *pCopy)

Copies memory between contexts.

###### Parameters

**pCopy**

  - Parameters for the memory copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Perform a 3D memory copy according to the parameters specified in pCopy. See the definition of the
CUDA_MEMCPY3D_PEER structure for documentation of its parameters.





See also:

cuMemcpyDtoD, cuMemcpyPeer, cuMemcpyDtoDAsync, cuMemcpyPeerAsync,
cuMemcpy3DPeerAsync, cudaMemcpy3DPeer