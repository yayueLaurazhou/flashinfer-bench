# CUresult cuMemFree (CUdeviceptr dptr)

Frees device memory.

###### Parameters

**dptr**

  - Pointer to memory to free

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Frees the memory space pointed to by dptr, which must have been returned by a previous
call to one of the following memory allocation APIs - cuMemAlloc(), cuMemAllocPitch(),
cuMemAllocManaged(), cuMemAllocAsync(), cuMemAllocFromPoolAsync()

Note - This API will not perform any implict synchronization when the pointer was allocated with
cuMemAllocAsync or cuMemAllocFromPoolAsync. Callers must ensure that all accesses to these
pointer have completed before invoking cuMemFree. For best performance and memory reuse, users
should use cuMemFreeAsync to free memory allocated via the stream ordered memory allocator. For
all other pointers, this API may perform implicit synchronization.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemAllocManaged, cuMemAllocAsync,
cuMemAllocFromPoolAsync, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemFreeAsync, cuMemHostGetDevicePointer, cuMemsetD2D8,
cuMemsetD2D16, cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaFree


CUDA Driver API TRM-06703-001 _vRelease Version  |  240


Modules