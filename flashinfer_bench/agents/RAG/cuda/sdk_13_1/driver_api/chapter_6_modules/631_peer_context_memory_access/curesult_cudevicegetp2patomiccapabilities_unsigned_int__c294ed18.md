# CUresult cuDeviceGetP2PAtomicCapabilities (unsigned int *capabilities, const CUatomicOperation *operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice)

Queries details about atomic operations supported between two devices.

###### Parameters

**capabilities**

  - Returned capability details of each requested operation
**operations**

  - Requested operations
**count**

  - Count of requested operations and size of capabilities
**srcDevice**

  - The source device of the target link
**dstDevice**

  - The destination device of the target link

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *capabilities the details about requested atomic *operations over the
the link between srcDevice and dstDevice. The allocated size of *operations and
*capabilities must be count.

For each CUatomicOperation in *operations, the corresponding result in *capabilities will
be a bitmask indicating which of CUatomicOperationCapability the link supports natively.

Returns CUDA_ERROR_INVALID_DEVICE if srcDevice or dstDevice are not valid or if they
represent the same device.

Returns CUDA_ERROR_INVALID_VALUE if *capabilities or *operations is NULL, if
count is 0, or if any of *operations is not valid.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  554


Modules


cuDeviceGetP2PAttribute, cudaDeviceGetP2PAttribute, cudaDeviceGetP2PAtomicCapabilities