# enum CUmemPool_attribute

CUDA memory pool attributes

###### Values

**CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1**
(value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another
streams as long as a stream ordering dependency of the allocating stream on the free action exists.
Cuda events and null stream interactions can create the required stream ordered dependencies.
(default enabled)
**CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC**
(value type = int) Allow reuse of already completed frees when there is no dependency between the
free and allocation. (default enabled)
**CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES**
(value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish
the stream ordering required to reuse a piece of memory released by cuMemFreeAsync (default
enabled).
**CU_MEMPOOL_ATTR_RELEASE_THRESHOLD**
(value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release
memory back to the OS. When more than the release threshold bytes of memory are held by the


CUDA Driver API TRM-06703-001 _vRelease Version  |  67


Modules


memory pool, the allocator will try to release memory back to the OS on the next call to stream,
event or context synchronize. (default 0)
**CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT**
(value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.
**CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH**
(value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the
last time it was reset. High watermark can only be reset to zero.
**CU_MEMPOOL_ATTR_USED_MEM_CURRENT**
(value type = cuuint64_t) Amount of memory from the pool that is currently in use by the
application.
**CU_MEMPOOL_ATTR_USED_MEM_HIGH**
(value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use
by the application since the last time it was reset. High watermark can only be reset to zero.