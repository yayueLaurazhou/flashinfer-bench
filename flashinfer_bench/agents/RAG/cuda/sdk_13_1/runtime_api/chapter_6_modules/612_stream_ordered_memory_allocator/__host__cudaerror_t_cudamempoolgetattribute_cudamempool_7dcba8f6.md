# __host__cudaError_t cudaMemPoolGetAttribute (cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value)

Gets attributes of a memory pool.

##### Parameters

**memPool**
**attr**

  - The attribute to get
**value**

  - Retrieved value

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Supported attributes are:


CUDA Runtime API vRelease Version  |  219


Modules


cudaMemPoolAttrReleaseThreshold: (value type = cuuint64_t) Amount of reserved memory in

##### **‣**

bytes to hold onto before trying to release memory back to the OS. When more than the release
threshold bytes of memory are held by the memory pool, the allocator will try to release memory
back to the OS on the next call to stream, event or context synchronize. (default 0)
cudaMemPoolReuseFollowEventDependencies: (value type = int) Allow cudaMallocAsync to use

##### **‣**

memory asynchronously freed in another stream as long as a stream ordering dependency of the
allocating stream on the free action exists. Cuda events and null stream interactions can create the
required stream ordered dependencies. (default enabled)
cudaMemPoolReuseAllowOpportunistic: (value type = int) Allow reuse of already completed frees

##### **‣**

when there is no dependency between the free and allocation. (default enabled)
cudaMemPoolReuseAllowInternalDependencies: (value type = int) Allow cudaMallocAsync to

##### **‣**

insert new stream dependencies in order to establish the stream ordering required to reuse a piece
of memory released by cudaFreeAsync (default enabled).
cudaMemPoolAttrReservedMemCurrent: (value type = cuuint64_t) Amount of backing memory

##### **‣**

currently allocated for the mempool.
cudaMemPoolAttrReservedMemHigh: (value type = cuuint64_t) High watermark of backing

##### **‣**

memory allocated for the mempool since the last time it was reset.
cudaMemPoolAttrUsedMemCurrent: (value type = cuuint64_t) Amount of memory from the pool

##### **‣**

that is currently in use by the application.
cudaMemPoolAttrUsedMemHigh: (value type = cuuint64_t) High watermark of the amount of

##### **‣**

memory from the pool that was in use by the application since the last time it was reset.





See also:

cuMemPoolGetAttribute, cudaMallocAsync, cudaFreeAsync, cudaDeviceGetDefaultMemPool,
cudaDeviceGetMemPool, cudaMemPoolCreate


CUDA Runtime API vRelease Version  |  220


Modules