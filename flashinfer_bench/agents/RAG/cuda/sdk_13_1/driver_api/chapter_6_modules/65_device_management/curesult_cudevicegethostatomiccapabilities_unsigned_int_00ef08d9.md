# CUresult cuDeviceGetHostAtomicCapabilities (unsigned int *capabilities, const CUatomicOperation *operations, unsigned int count, CUdevice dev)

Queries details about atomic operations supported between the device and host.

###### Parameters

**capabilities**

  - Returned capability details of each requested operation
**operations**

  - Requested operations
**count**

  - Count of requested operations and size of capabilities
**dev**

  - Device handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *capabilities the details about requested atomic *operations over the the link
between dev and the host. The allocated size of *operations and *capabilities must be
count.


CUDA Driver API TRM-06703-001 _vRelease Version  |  104


Modules


For each CUatomicOperation in *operations, the corresponding result in *capabilities will
be a bitmask indicating which of CUatomicOperationCapability the link supports natively.

Returns CUDA_ERROR_INVALID_DEVICE if dev is not valid.

Returns CUDA_ERROR_INVALID_VALUE if *capabilities or *operations is NULL, if
count is 0, or if any of *operations is not valid.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetP2PAtomicCapabilities, cudaDeviceGeHostAtomicCapabilities