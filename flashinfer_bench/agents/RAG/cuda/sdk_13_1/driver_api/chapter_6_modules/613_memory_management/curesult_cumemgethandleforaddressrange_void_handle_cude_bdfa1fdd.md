# CUresult cuMemGetHandleForAddressRange (void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags)

Retrieve handle for an address range.

###### Parameters

**handle**

  - Pointer to the location where the returned handle will be stored.
**dptr**

  - Pointer to a valid CUDA device allocation. Must be aligned to host page size.
**size**

  - Length of the address range. Must be aligned to host page size.


CUDA Driver API TRM-06703-001 _vRelease Version  |  242


Modules


**handleType**

  - Type of handle requested (defines type and size of the handle output parameter)
**flags**

  - When requesting
CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD the value
could be CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE, otherwise 0.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Get a handle of the specified type to an address range. When requesting
CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, address
range obtained by a prior call to either cuMemAlloc or cuMemAddressReserve is supported if the
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED device attribute returns true. If the address
range was obtained via cuMemAddressReserve, it must also be fully mapped via cuMemMap. Address
range obtained by a prior call to either cuMemAllocHost or cuMemHostAlloc is supported if the
CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED device attribute returns
true.

As of CUDA 13.0, querying support for address range obtained by calling cuMemAllocHost or
cuMemHostAlloc using the CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED device attribute
is deprecated.

Users must ensure the dptr and size are aligned to the host page size.

The handle will be interpreted as a pointer to an integer to store the dma_buf file descriptor. Users
must ensure the entire address range is backed and mapped when the address range is allocated by
cuMemAddressReserve. All the physical allocations backing the address range must be resident on the
same device and have identical allocation properties. Users are also expected to retrieve a new handle
every time the underlying physical allocation(s) corresponding to a previously queried VA range are
changed.

For CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, users may
set flags to CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE. Which when set on
a supported platform, will give a DMA_BUF handle mapped via PCIE BAR1 or will return an error
otherwise.