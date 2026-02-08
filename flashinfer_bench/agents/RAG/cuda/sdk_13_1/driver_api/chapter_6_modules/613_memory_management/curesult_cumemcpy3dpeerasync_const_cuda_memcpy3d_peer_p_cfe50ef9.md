# CUresult cuMemcpy3DPeerAsync (const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)

Copies memory between contexts asynchronously.

###### Parameters

**pCopy**

  - Parameters for the memory copy
**hStream**

  - Stream identifier

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  219


Modules

###### Description

Perform a 3D memory copy according to the parameters specified in pCopy. See the definition of the
CUDA_MEMCPY3D_PEER structure for documentation of its parameters.











See also:

cuMemcpyDtoD, cuMemcpyPeer, cuMemcpyDtoDAsync, cuMemcpyPeerAsync,
cuMemcpy3DPeerAsync, cudaMemcpy3DPeerAsync