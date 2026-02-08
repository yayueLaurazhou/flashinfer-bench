# CUresult cuMemAlloc (CUdeviceptr *dptr, size_t bytesize)

Allocates device memory.

###### Parameters

**dptr**

  - Returned device pointer
**bytesize**

  - Requested allocation size in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Allocates bytesize bytes of linear memory on the device and returns in *dptr a pointer to the
allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory is
not cleared. If bytesize is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemcpyDtoH,
cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,


CUDA Driver API TRM-06703-001 _vRelease Version  |  195


Modules


cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMalloc