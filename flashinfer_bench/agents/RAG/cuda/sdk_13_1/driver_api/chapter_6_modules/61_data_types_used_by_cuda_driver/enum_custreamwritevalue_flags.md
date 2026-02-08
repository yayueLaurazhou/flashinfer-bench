# enum CUstreamWriteValue_flags

Flags for cuStreamWriteValue32

###### Values

**CU_STREAM_WRITE_VALUE_DEFAULT = 0x0**
Default behavior
**CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1**
Permits the write to be reordered with writes which were issued before it, as a performance
optimization. Normally, cuStreamWriteValue32 will provide a memory fence before the write,
which has similar semantics to __threadfence_system() but is scoped to the stream rather than a
CUDA thread. This flag is not supported in the v2 API.


CUDA Driver API TRM-06703-001 _vRelease Version  |  83


Modules