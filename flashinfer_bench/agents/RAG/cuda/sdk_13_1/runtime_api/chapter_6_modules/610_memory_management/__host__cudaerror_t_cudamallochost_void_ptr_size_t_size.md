# __host__cudaError_t cudaMallocHost (void **ptr, size_t size)

Allocates page-locked memory on the host.

##### Parameters

**ptr**

  - Pointer to allocated host memory
**size**

  - Requested allocation size in bytes

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Allocates size bytes of host memory that is page-locked and accessible to the device. The driver
tracks the virtual memory ranges allocated with this function and automatically accelerates calls to
functions such as cudaMemcpy*(). Since the memory can be accessed directly by the device, it can be
read or written with much higher bandwidth than pageable memory obtained with functions such as
malloc().

On systems where pageableMemoryAccessUsesHostPageTables is true, cudaMallocHost may not
page-lock the allocated memory.

Page-locking excessive amounts of memory with cudaMallocHost() may degrade system performance,
since it reduces the amount of memory available to the system for paging. As a result, this function is
best used sparingly to allocate staging areas for data exchange between host and device.





CUDA Runtime API vRelease Version  |  140


Modules


See also:

cudaMalloc, cudaMallocPitch, cudaMallocArray, cudaMalloc3D, cudaMalloc3DArray,
cudaHostAlloc, cudaFree, cudaFreeArray, cudaMallocHost ( C++ API), cudaFreeHost, cudaHostAlloc,
cuMemAllocHost