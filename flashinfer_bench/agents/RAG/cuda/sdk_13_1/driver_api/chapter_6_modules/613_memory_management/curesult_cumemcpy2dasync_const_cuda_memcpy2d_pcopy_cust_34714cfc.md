# CUresult cuMemcpy2DAsync (const CUDA_MEMCPY2D *pCopy, CUstream hStream)

Copies memory for 2D arrays.

###### Parameters

**pCopy**

  - Parameters for the memory copy
**hStream**

  - Stream identifier

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Perform a 2D memory copy according to the parameters specified in pCopy. The
CUDA_MEMCPY2D structure is defined as:


srcMemoryType and dstMemoryType specify the type of memory of the source and destination,

###### **‣**

respectively; CUmemorytype_enum is defined as:


If srcMemoryType is CU_MEMORYTYPE_HOST, srcHost and srcPitch specify the (host) base
address of the source data and the bytes per row to apply. srcArray is ignored.


CUDA Driver API TRM-06703-001 _vRelease Version  |  206


Modules


If srcMemoryType is CU_MEMORYTYPE_UNIFIED, srcDevice and srcPitch specify the (unified
virtual address space) base address of the source data and the bytes per row to apply. srcArray is
ignored. This value may be used only if unified addressing is supported in the calling context.


If srcMemoryType is CU_MEMORYTYPE_DEVICE, srcDevice and srcPitch specify the (device)
base address of the source data and the bytes per row to apply. srcArray is ignored.


If srcMemoryType is CU_MEMORYTYPE_ARRAY, srcArray specifies the handle of the source data.
srcHost, srcDevice and srcPitch are ignored.


If dstMemoryType is CU_MEMORYTYPE_UNIFIED, dstDevice and dstPitch specify the (unified
virtual address space) base address of the source data and the bytes per row to apply. dstArray is
ignored. This value may be used only if unified addressing is supported in the calling context.


If dstMemoryType is CU_MEMORYTYPE_HOST, dstHost and dstPitch specify the (host) base
address of the destination data and the bytes per row to apply. dstArray is ignored.


If dstMemoryType is CU_MEMORYTYPE_DEVICE, dstDevice and dstPitch specify the (device)
base address of the destination data and the bytes per row to apply. dstArray is ignored.


If dstMemoryType is CU_MEMORYTYPE_ARRAY, dstArray specifies the handle of the destination
data. dstHost, dstDevice and dstPitch are ignored.

srcXInBytes and srcY specify the base address of the source data for the copy.

###### **‣**

For host pointers, the starting address is
‎ void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);

For device pointers, the starting address is
‎ CUdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

dstXInBytes and dstY specify the base address of the destination data for the copy.

###### **‣**

For host pointers, the base address is
‎ void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);

For device pointers, the starting address is
‎ CUdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.

###### **‣**

If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch

###### **‣**

must be greater than or equal to WidthInBytes + dstXInBytes.
If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch

###### **‣**

must be greater than or equal to WidthInBytes + dstXInBytes.


CUDA Driver API TRM-06703-001 _vRelease Version  |  207


Modules


If specified, srcHeight must be greater than or equal to Height + srcY, and dstHeight must be

###### **‣**

greater than or equal to Height + dstY.

cuMemcpy2DAsync() returns an error if any pitch is greater than the maximum allowed
(CU_DEVICE_ATTRIBUTE_MAX_PITCH). cuMemAllocPitch() passes back pitches that always
work with cuMemcpy2D(). On intra-device memory copies (device to device, CUDA array to
device, CUDA array to CUDA array), cuMemcpy2DAsync() may fail for pitches not computed by
cuMemAllocPitch().











See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D8Async,
cuMemsetD2D16, cuMemsetD2D16Async, cuMemsetD2D32, cuMemsetD2D32Async, cuMemsetD8,
cuMemsetD8Async, cuMemsetD16, cuMemsetD16Async, cuMemsetD32, cuMemsetD32Async,
cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync