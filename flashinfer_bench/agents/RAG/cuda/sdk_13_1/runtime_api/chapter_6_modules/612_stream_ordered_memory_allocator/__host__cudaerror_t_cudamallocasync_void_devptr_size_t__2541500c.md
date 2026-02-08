# __host__cudaError_t cudaMallocAsync (void **devPtr, size_t size, cudaStream_t hStream)

Allocates memory with stream ordered semantics.

##### Parameters

**devPtr**

  - Returned device pointer
**size**

  - Number of bytes to allocate
**hStream**

  - The stream establishing the stream ordering contract and the memory pool to allocate from

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, cudaErrorOutOfMemory,

##### Description

Inserts an allocation operation into hStream. A pointer to the allocated memory is returned
immediately in *dptr. The allocation must not be accessed until the the allocation operation completes.
The allocation comes from the memory pool associated with the stream's device.


Note:

**‣** The default memory pool of a device contains device memory from that device.

**‣** Basic stream ordering allows future work submitted into the same stream to use the allocation.
Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
operation completes before work submitted in a separate stream runs.

**‣** During stream capture, this function results in the creation of an allocation node. In this case, the
allocation is owned by the graph instead of the memory pool. The memory pool's properties are used
to set the node's creation parameters.











CUDA Runtime API vRelease Version  |  212


Modules


See also:

cuMemAllocAsync, cudaMallocAsync ( C++ API), cudaMallocFromPoolAsync, cudaFreeAsync,
cudaDeviceSetMemPool, cudaDeviceGetDefaultMemPool, cudaDeviceGetMemPool,
cudaMemPoolSetAccess, cudaMemPoolSetAttribute, cudaMemPoolGetAttribute