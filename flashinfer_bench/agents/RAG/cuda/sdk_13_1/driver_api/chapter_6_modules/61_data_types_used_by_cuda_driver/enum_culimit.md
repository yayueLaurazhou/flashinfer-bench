# enum CUlimit

Limits

###### Values

**CU_LIMIT_STACK_SIZE = 0x00**
GPU thread stack size
**CU_LIMIT_PRINTF_FIFO_SIZE = 0x01**
GPU printf FIFO size
**CU_LIMIT_MALLOC_HEAP_SIZE = 0x02**
GPU malloc heap size
**CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03**
GPU device runtime launch synchronize depth
**CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04**
GPU device runtime pending launch count
**CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05**
A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a
hint
**CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06**
A size in bytes for L2 persisting lines cache size
**CU_LIMIT_SHMEM_SIZE = 0x07**
A maximum size in bytes of shared memory available to CUDA kernels on a CIG context. Can only
be queried, cannot be set
**CU_LIMIT_CIG_ENABLED = 0x08**
A non-zero value indicates this CUDA context is a CIG-enabled context. Can only be queried,
cannot be set
**CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED = 0x09**
When set to zero, CUDA will fail to launch a kernel on a CIG context, instead of using the fallback
path, if the kernel uses more shared memory than available
**CU_LIMIT_MAX**


CUDA Driver API TRM-06703-001 _vRelease Version  |  62


Modules