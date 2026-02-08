# __host__cudaError_t cudaMemPoolSetAttribute (cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value)

Sets attributes of a memory pool.

##### Parameters

**memPool**
**attr**

  - The attribute to modify
**value**

  - Pointer to the value to assign

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Supported attributes are:

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


CUDA Runtime API vRelease Version  |  223


Modules


cudaMemPoolAttrReservedMemHigh: (value type = cuuint64_t) Reset the high watermark that

##### **‣**

tracks the amount of backing memory that was allocated for the memory pool. It is illegal to set
this attribute to a non-zero value.
cudaMemPoolAttrUsedMemHigh: (value type = cuuint64_t) Reset the high watermark that tracks

##### **‣**

the amount of used memory that was allocated for the memory pool. It is illegal to set this attribute
to a non-zero value.





See also:

cuMemPoolSetAttribute, cudaMallocAsync, cudaFreeAsync, cudaDeviceGetDefaultMemPool,
cudaDeviceGetMemPool, cudaMemPoolCreate