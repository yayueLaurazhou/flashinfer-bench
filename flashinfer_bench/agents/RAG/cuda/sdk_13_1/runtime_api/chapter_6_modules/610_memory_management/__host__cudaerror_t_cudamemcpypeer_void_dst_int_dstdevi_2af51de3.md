# __host__cudaError_t cudaMemcpyPeer (void *dst, int dstDevice, const void *src, int srcDevice, size_t count)

Copies memory between two devices.

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

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Copies memory from one device to memory on another device. dst is the base device pointer of the
destination memory and dstDevice is the destination device. src is the base device pointer of the
source memory and srcDevice is the source device. count specifies the number of bytes to copy.


CUDA Runtime API vRelease Version  |  177


Modules


Note that this function is asynchronous with respect to the host, but serialized with respect all
pending and future asynchronous work in to the current device, srcDevice, and dstDevice (use
cudaMemcpyPeerAsync to avoid this synchronization).











See also:

cudaMemcpy, cudaMemcpyAsync, cudaMemcpyPeerAsync, cudaMemcpy3DPeerAsync,
cuMemcpyPeer