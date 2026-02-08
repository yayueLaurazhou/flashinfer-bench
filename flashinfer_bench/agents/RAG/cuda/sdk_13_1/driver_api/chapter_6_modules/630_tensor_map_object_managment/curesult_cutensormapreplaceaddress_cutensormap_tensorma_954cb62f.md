# CUresult cuTensorMapReplaceAddress (CUtensorMap *tensorMap, void *globalAddress)

Modify an existing tensor map descriptor with an updated global address.

###### Parameters

**tensorMap**

  - Tensor map object to modify
**globalAddress**

  - Starting address of memory region described by tensor, must follow previous alignment
requirements

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  550


Modules

###### Description

Modifies the descriptor for Tensor Memory Access (TMA) object passed in tensorMap with an
updated globalAddress.

Tensor map objects are only supported on devices of compute capability 9.0 or higher. Additionally, a
tensor map object is an opaque value, and, as such, should only be accessed through CUDA API calls.


See also:

cuTensorMapEncodeTiled, cuTensorMapEncodeIm2col, cuTensorMapEncodeIm2colWide