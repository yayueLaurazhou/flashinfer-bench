# CUresult cuMemHostGetDevicePointer (CUdeviceptr *pdptr, void *p, unsigned int Flags)

Passes back device pointer of mapped pinned memory.

###### Parameters

**pdptr**

  - Returned device pointer
**p**

  - Host pointer
**Flags**

  - Options (must be 0)

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  246


Modules

###### Description

Passes back the device pointer pdptr corresponding to the mapped, pinned host buffer p allocated by
cuMemHostAlloc.

cuMemHostGetDevicePointer() will fail if the CU_MEMHOSTALLOC_DEVICEMAP flag was not
specified at the time the memory was allocated, or if the function is called on a GPU that does not
support mapped pinned memory.

For devices that have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the
memory can also be accessed from the device using the host pointer p. The device pointer returned
by cuMemHostGetDevicePointer() may or may not match the original host pointer p and depends on
the devices visible to the application. If all devices visible to the application have a non-zero value
for the device attribute, the device pointer returned by cuMemHostGetDevicePointer() will match the
original pointer p. If any device visible to the application has a zero value for the device attribute, the
device pointer returned by cuMemHostGetDevicePointer() will not match the original host pointer p,
but it will be suitable for use on all devices provided Unified Virtual Addressing is enabled. In such
systems, it is valid to access the memory using either pointer on devices that have a non-zero value for
the device attribute. Note however that such devices should access the memory using only one of the
two pointers and not both.

Flags provides for future releases. For now, it must be set to 0.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD,
cuMemcpyAtoH, cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemsetD2D8, cuMemsetD2D16, cuMemsetD2D32, cuMemsetD8,
cuMemsetD16, cuMemsetD32, cudaHostGetDevicePointer


CUDA Driver API TRM-06703-001 _vRelease Version  |  247


Modules