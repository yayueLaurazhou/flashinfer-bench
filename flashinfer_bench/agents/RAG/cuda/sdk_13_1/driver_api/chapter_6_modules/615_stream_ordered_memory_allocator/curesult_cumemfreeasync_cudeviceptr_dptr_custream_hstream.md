# CUresult cuMemFreeAsync (CUdeviceptr dptr, CUstream hStream)

Frees memory with stream ordered semantics.

###### Parameters

**dptr**

  - memory to free
**hStream**

  - The stream establishing the stream ordering contract.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT (default stream specified with no current context),
CUDA_ERROR_NOT_SUPPORTED

###### Description

Inserts a free operation into hStream. The allocation must not be accessed after stream execution
reaches the free. After this API returns, accessing the memory from any subsequent work launched on
the GPU or querying its pointer attributes results in undefined behavior.


Note:


During stream capture, this function results in the creation of a free node and must therefore be passed
the address of a graph allocation.