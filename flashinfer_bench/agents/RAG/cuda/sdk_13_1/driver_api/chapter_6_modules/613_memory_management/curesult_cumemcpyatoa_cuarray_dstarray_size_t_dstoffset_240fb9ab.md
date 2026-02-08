# CUresult cuMemcpyAtoA (CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)

Copies memory from Array to Array.

###### Parameters

**dstArray**

  - Destination array
**dstOffset**

  - Offset in bytes of destination array
**srcArray**

  - Source array
**srcOffset**

  - Offset in bytes of source array
**ByteCount**

  - Size of memory copy in bytes


CUDA Driver API TRM-06703-001 _vRelease Version  |  221


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Copies from one 1D CUDA array to another. dstArray and srcArray specify the handles of
the destination and source CUDA arrays for the copy, respectively. dstOffset and srcOffset
specify the destination and source offsets in bytes into the CUDA arrays. ByteCount is the number
of bytes to be copied. The size of the elements in the CUDA arrays need not be the same format, but
the elements must be the same size; and count must be evenly divisible by that size.





See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMemcpyArrayToArray