# CUresult cuMemAllocPitch (CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)

Allocates pitched device memory.

###### Parameters

**dptr**

  - Returned device pointer
**pPitch**

  - Returned pitch of allocation in bytes


CUDA Driver API TRM-06703-001 _vRelease Version  |  199


Modules


**WidthInBytes**

  - Requested allocation width in bytes
**Height**

  - Requested allocation height in rows
**ElementSizeBytes**

  - Size of largest reads/writes for range

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Allocates at least WidthInBytes * Height bytes of linear memory on the device and returns
in *dptr a pointer to the allocated memory. The function may pad the allocation to ensure that
corresponding pointers in any given row will continue to meet the alignment requirements for
coalescing as the address is updated from row to row. ElementSizeBytes specifies the size of
the largest reads and writes that will be performed on the memory range. ElementSizeBytes
may be 4, 8 or 16 (since coalesced memory transactions are not possible on other data sizes). If
ElementSizeBytes is smaller than the actual read/write size of a kernel, the kernel will run
correctly, but possibly at reduced speed. The pitch returned in *pPitch by cuMemAllocPitch() is the
width in bytes of the allocation. The intended usage of pitch is as a separate parameter of the allocation,
used to compute addresses within the 2D array. Given the row and column of an array element of type
T, the address is computed as:
‎  T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;

The pitch returned by cuMemAllocPitch() is guaranteed to work with cuMemcpy2D() under all
circumstances. For allocations of 2D arrays, it is recommended that programmers consider performing
pitch allocations using cuMemAllocPitch(). Due to alignment restrictions in the hardware, this is
especially true if the application will be performing 2D memory copies between different regions of
device memory (whether linear memory or CUDA arrays).

The byte alignment of the pitch returned by cuMemAllocPitch() is guaranteed to match or exceed the
alignment requirement for texture binding with cuTexRefSetAddress2D().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,


CUDA Driver API TRM-06703-001 _vRelease Version  |  200


Modules


cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemcpyDtoH,
cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMallocPitch