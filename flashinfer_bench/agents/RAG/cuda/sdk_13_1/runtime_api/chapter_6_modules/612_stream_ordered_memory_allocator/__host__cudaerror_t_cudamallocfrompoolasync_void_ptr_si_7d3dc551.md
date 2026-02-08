# __host__cudaError_t cudaMallocFromPoolAsync (void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)

Allocates memory from a specified pool with stream ordered semantics.

##### Parameters

**ptr**

  - Returned device pointer
**size**
**memPool**

  - The pool to allocate from
**stream**

  - The stream establishing the stream ordering semantic

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, cudaErrorOutOfMemory

##### Description

Inserts an allocation operation into hStream. A pointer to the allocated memory is returned
immediately in *dptr. The allocation must not be accessed until the the allocation operation completes.
The allocation comes from the specified memory pool.


Note:

**‣** The specified memory pool may be from a device different than that of the specified hStream.


Basic stream ordering allows future work submitted into the same stream to use the allocation.

##### **‣**

Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
operation completes before work submitted in a separate stream runs.


Note:


During stream capture, this function results in the creation of an allocation node. In this case, the
allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to
set the node's creation parameters.


CUDA Runtime API vRelease Version  |  213


Modules


See also:

cuMemAllocFromPoolAsync, cudaMallocAsync ( C++ API), cudaMallocAsync, cudaFreeAsync,
cudaDeviceGetDefaultMemPool, cudaMemPoolCreate, cudaMemPoolSetAccess,
cudaMemPoolSetAttribute