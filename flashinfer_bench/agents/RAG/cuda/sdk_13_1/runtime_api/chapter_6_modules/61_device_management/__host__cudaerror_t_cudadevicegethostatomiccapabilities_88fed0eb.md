# __host__cudaError_t cudaDeviceGetHostAtomicCapabilities (unsigned int *capabilities, const cudaAtomicOperation *operations, unsigned int count, int device)

Queries details about atomic operations supported between the device and host.

##### Parameters

**capabilities**

  - Returned capability details of each requested operation
**operations**

  - Requested operations
**count**

  - Count of requested operations and size of capabilities
**device**

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

##### Description

Returns in *capabilities the details about requested atomic *operations over the the link
between dev and the host. The allocated size of *operations and *capabilities must be
count.


CUDA Runtime API vRelease Version  |  15


Modules


For each cudaAtomicOperation in *operations, the corresponding result in *capabilities will
be a bitmask indicating which of cudaAtomicOperationCapability the link supports natively.

Returns cudaErrorInvalidDevice if dev is not valid.

Returns cudaErrorInvalidValue if *capabilities or *operations is NULL, if count is 0, or if
any of *operations is not valid.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaDeviceGetAttribute, cudaDeviceGetP2PAtomicCapabilities, cuDeviceGeHostAtomicCapabilities