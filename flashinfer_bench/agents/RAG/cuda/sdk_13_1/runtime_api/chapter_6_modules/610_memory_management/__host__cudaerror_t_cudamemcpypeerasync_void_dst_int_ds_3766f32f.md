# __host__cudaError_t cudaMemcpyPeerAsync (void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream)

Copies memory between two devices asynchronously.

##### Parameters

**dst**

  - Destination device pointer
**dstDevice**

  - Destination device
**src**

  - Source device pointer
**srcDevice**

  - Source device
**count**

  - Size of memory copy in bytes
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice


CUDA Runtime API vRelease Version  |  178


Modules

##### Description

Copies memory from one device to memory on another device. dst is the base device pointer of the
destination memory and dstDevice is the destination device. src is the base device pointer of the
source memory and srcDevice is the source device. count specifies the number of bytes to copy.

Note that this function is asynchronous with respect to the host and all work on other devices.













See also:

cudaMemcpy, cudaMemcpyPeer, cudaMemcpyAsync, cudaMemcpy3DPeerAsync,
cuMemcpyPeerAsync