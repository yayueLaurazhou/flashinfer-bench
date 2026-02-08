# enum cudaMemPoolAttr

CUDA memory pool attributes

##### Values

**cudaMemPoolReuseFollowEventDependencies = 0x1**
(value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another
streams as long as a stream ordering dependency of the allocating stream on the free action exists.
Cuda events and null stream interactions can create the required stream ordered dependencies.
(default enabled)
**cudaMemPoolReuseAllowOpportunistic = 0x2**
(value type = int) Allow reuse of already completed frees when there is no dependency between the
free and allocation. (default enabled)
**cudaMemPoolReuseAllowInternalDependencies = 0x3**
(value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish
the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled).
**cudaMemPoolAttrReleaseThreshold = 0x4**
(value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release
memory back to the OS. When more than the release threshold bytes of memory are held by the
memory pool, the allocator will try to release memory back to the OS on the next call to stream,
event or context synchronize. (default 0)
**cudaMemPoolAttrReservedMemCurrent = 0x5**
(value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.
**cudaMemPoolAttrReservedMemHigh = 0x6**
(value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the
last time it was reset. High watermark can only be reset to zero.
**cudaMemPoolAttrUsedMemCurrent = 0x7**
(value type = cuuint64_t) Amount of memory from the pool that is currently in use by the
application.
**cudaMemPoolAttrUsedMemHigh = 0x8**
(value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use
by the application since the last time it was reset. High watermark can only be reset to zero.