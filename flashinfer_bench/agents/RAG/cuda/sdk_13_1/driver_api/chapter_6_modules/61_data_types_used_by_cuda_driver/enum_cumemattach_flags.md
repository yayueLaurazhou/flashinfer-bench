# enum CUmemAttach_flags

CUDA Mem Attach Flags

###### Values

**CU_MEM_ATTACH_GLOBAL = 0x1**
Memory can be accessed by any stream on any device
**CU_MEM_ATTACH_HOST = 0x2**
Memory cannot be accessed by any stream on any device
**CU_MEM_ATTACH_SINGLE = 0x4**
Memory can only be accessed by a single stream on the associated device