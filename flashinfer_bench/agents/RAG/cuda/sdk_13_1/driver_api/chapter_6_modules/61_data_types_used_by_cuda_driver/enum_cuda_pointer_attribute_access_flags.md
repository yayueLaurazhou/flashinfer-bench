# enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS

Access flags that specify the level of access the current context's device has on the memory referenced.

###### Values

**CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0x0**
No access, meaning the device cannot access this memory at all, thus must be staged through
accessible memory in order to complete certain operations
**CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 0x1**
Read-only access, meaning writes to this memory are considered invalid accesses and thus return
error in that case.
**CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 0x3**
Read-write access, the device has full read-write access to the memory