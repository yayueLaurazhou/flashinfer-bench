# CUresult cuArrayCreate (CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)

Creates a 1D or 2D CUDA array.

###### Parameters

**pHandle**

  - Returned array
**pAllocateArray**

  - Array descriptor


CUDA Driver API TRM-06703-001 _vRelease Version  |  180


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN

###### Description

Creates a CUDA array according to the CUDA_ARRAY_DESCRIPTOR structure
pAllocateArray and returns a handle to the new CUDA array in *pHandle. The
CUDA_ARRAY_DESCRIPTOR is defined as:


, and are the width, and height of the CUDA array (in elements); the CUDA array

###### ‣ Width Height

is one-dimensional if height is 0, two-dimensional otherwise;
Format specifies the format of the elements; CUarray_format is defined as:

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  181


Modules


specifies the number of packed components per CUDA array element; it may be
1, 2, or 4;

Here are examples of CUDA array descriptions:

Description for a CUDA array of 2048 floats:


Description for a 64 x 64 CUDA array of floats:


Description for a width x height CUDA array of 16-bit elements, each of which is two 8-bit
unsigned chars:


CUDA Driver API TRM-06703-001 _vRelease Version  |  182


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD,
cuMemcpyAtoH, cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMallocArray