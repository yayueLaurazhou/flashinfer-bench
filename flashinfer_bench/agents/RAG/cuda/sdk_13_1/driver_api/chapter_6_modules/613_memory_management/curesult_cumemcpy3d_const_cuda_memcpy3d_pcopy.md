# CUresult cuMemcpy3D (const CUDA_MEMCPY3D *pCopy)

Copies memory for 3D arrays.

###### Parameters

**pCopy**

  - Parameters for the memory copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Perform a 3D memory copy according to the parameters specified in pCopy. The
CUDA_MEMCPY3D structure is defined as:


CUDA Driver API TRM-06703-001 _vRelease Version  |  211


Modules


srcMemoryType and dstMemoryType specify the type of memory of the source and destination,

###### **‣**

respectively; CUmemorytype_enum is defined as:


If srcMemoryType is CU_MEMORYTYPE_UNIFIED, srcDevice and srcPitch specify the (unified
virtual address space) base address of the source data and the bytes per row to apply. srcArray is
ignored. This value may be used only if unified addressing is supported in the calling context.


If srcMemoryType is CU_MEMORYTYPE_HOST, srcHost, srcPitch and srcHeight specify the (host)
base address of the source data, the bytes per row, and the height of each 2D slice of the 3D array.
srcArray is ignored.


If srcMemoryType is CU_MEMORYTYPE_DEVICE, srcDevice, srcPitch and srcHeight specify the
(device) base address of the source data, the bytes per row, and the height of each 2D slice of the 3D
array. srcArray is ignored.


If srcMemoryType is CU_MEMORYTYPE_ARRAY, srcArray specifies the handle of the source data.
srcHost, srcDevice, srcPitch and srcHeight are ignored.


If dstMemoryType is CU_MEMORYTYPE_UNIFIED, dstDevice and dstPitch specify the (unified
virtual address space) base address of the source data and the bytes per row to apply. dstArray is
ignored. This value may be used only if unified addressing is supported in the calling context.


If dstMemoryType is CU_MEMORYTYPE_HOST, dstHost and dstPitch specify the (host) base
address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array.
dstArray is ignored.


If dstMemoryType is CU_MEMORYTYPE_DEVICE, dstDevice and dstPitch specify the (device)
base address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array.
dstArray is ignored.


If dstMemoryType is CU_MEMORYTYPE_ARRAY, dstArray specifies the handle of the destination
data. dstHost, dstDevice, dstPitch and dstHeight are ignored.

srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.

###### **‣**

For host pointers, the starting address is
‎ void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch +
srcXInBytes);

For device pointers, the starting address is
‎ CUdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.


CUDA Driver API TRM-06703-001 _vRelease Version  |  212


Modules


dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.

###### **‣**

For host pointers, the base address is
‎ void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch +
dstXInBytes);

For device pointers, the starting address is
‎ CUdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy

###### **‣**

being performed.
If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch

###### **‣**

must be greater than or equal to WidthInBytes + dstXInBytes.
If specified, srcHeight must be greater than or equal to Height + srcY, and dstHeight must be

###### **‣**

greater than or equal to Height + dstY.

cuMemcpy3D() returns an error if any pitch is greater than the maximum allowed
(CU_DEVICE_ATTRIBUTE_MAX_PITCH).

The srcLOD and dstLOD members of the CUDA_MEMCPY3D structure must be set to 0.





See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMemcpy3D


CUDA Driver API TRM-06703-001 _vRelease Version  |  213


Modules