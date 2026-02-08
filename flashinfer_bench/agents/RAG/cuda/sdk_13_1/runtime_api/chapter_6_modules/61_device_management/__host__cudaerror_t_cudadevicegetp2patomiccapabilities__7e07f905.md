# __host__cudaError_t cudaDeviceGetP2PAtomicCapabilities (unsigned int *capabilities, const cudaAtomicOperation *operations, unsigned int count, int srcDevice, int dstDevice)

Queries details about atomic operations supported between two devices.

##### Parameters

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

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

##### Description

Returns in *capabilities the details about requested atomic *operations over the
the link between srcDevice and dstDevice. The allocated size of *operations and
*capabilities must be count.

For each cudaAtomicOperation in *operations, the corresponding result in *capabilities will
be a bitmask indicating which of cudaAtomicOperationCapability the link supports natively.


CUDA Runtime API vRelease Version  |  19


Modules


Returns cudaErrorInvalidDevice if srcDevice or dstDevice are not valid or if they represent the
same device.

Returns cudaErrorInvalidValue if *capabilities or *operations is NULL, if count is 0, or if
any of *operations is not valid.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaDeviceGetP2PAttribute, cuDeviceGetP2PAttribute, cuDeviceGetP2PAtomicCapabilities