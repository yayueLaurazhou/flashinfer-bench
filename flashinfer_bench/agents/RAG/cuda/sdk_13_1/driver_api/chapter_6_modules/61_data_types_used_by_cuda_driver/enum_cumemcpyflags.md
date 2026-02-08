# enum CUmemcpyFlags

Flags to specify for copies within a batch. For more details see cuMemcpyBatchAsync.

###### Values

**CU_MEMCPY_FLAG_DEFAULT = 0x0**
**CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 0x1**
Hint to the driver to try and overlap the copy with compute work on the SMs.