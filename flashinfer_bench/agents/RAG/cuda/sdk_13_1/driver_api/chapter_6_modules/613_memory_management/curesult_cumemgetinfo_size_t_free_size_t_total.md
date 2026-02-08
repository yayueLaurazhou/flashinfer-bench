# CUresult cuMemGetInfo (size_t *free, size_t *total)

Gets free and total memory.

###### Parameters

**free**

  - Returned free memory in bytes


CUDA Driver API TRM-06703-001 _vRelease Version  |  243


Modules


**total**

  - Returned total memory in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *total the total amount of memory available to the the current context. Returns in
*free the amount of memory on the device that is free according to the OS. CUDA is not guaranteed
to be able to allocate all of the memory that the OS reports as free. In a multi-tenet situation, free
estimate returned is prone to race condition where a new allocation/free done by a different process or
a different thread in the same process between the time when free memory was estimated and reported,
will result in deviation in free value reported and actual free memory.

The integrated GPU on Tegra shares memory with CPU and other component of the SoC. The free and
total values returned by the API excludes the SWAP memory space maintained by the OS on some
platforms. The OS may move some of the memory pages into swap area as the GPU or CPU allocate or
access memory. See Tegra app note on how to calculate total and free memory on Tegra.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD,
cuMemcpyAtoH, cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemHostAlloc,
cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16, cuMemsetD2D32, cuMemsetD8,
cuMemsetD16, cuMemsetD32, cudaMemGetInfo