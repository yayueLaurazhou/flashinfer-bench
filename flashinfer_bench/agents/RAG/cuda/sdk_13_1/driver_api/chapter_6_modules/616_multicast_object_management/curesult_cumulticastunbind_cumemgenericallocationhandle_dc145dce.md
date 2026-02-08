# CUresult cuMulticastUnbind (CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size)

Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.

###### Parameters

**mcHandle**
Handle representing a multicast object.
**dev**
Device that hosts the memory allocation.
**mcOffset**
Offset into the multicast object.
**size**
Desired size to unbind.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

Unbinds any memory allocations hosted on dev and bound to a multicast object at mcOffset
and upto a given size. The intended size of the unbind and the offset in the multicast range
( mcOffset ) must be a multiple of the value returned by cuMulticastGetGranularity flag


CUDA Driver API TRM-06703-001 _vRelease Version  |  310


Modules


CU_MULTICAST_GRANULARITY_MINIMUM. The size + mcOffset cannot be larger than the
total size of the multicast object.


Note:

Warning: The mcOffset and the size must match the corresponding values specified during the
bind call. Any other values may result in undefined behavior.


See also:

cuMulticastBindMem, cuMulticastBindAddr

cuMulticastBindMem_v2, cuMulticastBindAddr_v2