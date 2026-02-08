# __host__cudaError_t cudaDeviceGraphMemTrim (int device)

Free unused memory that was cached on the specified device for use with graphs back to the OS.

##### Parameters

**device**

  - The device for which cached memory should be freed.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Blocks which are not in use by a graph that is either currently executing or scheduled to execute are
freed back to the operating system.









CUDA Runtime API vRelease Version  |  321


Modules





See also:

cudaGraphAddMemAllocNode, cudaGraphAddMemFreeNode, cudaDeviceGetGraphMemAttribute,
cudaDeviceSetGraphMemAttribute, cudaMallocAsync, cudaFreeAsync