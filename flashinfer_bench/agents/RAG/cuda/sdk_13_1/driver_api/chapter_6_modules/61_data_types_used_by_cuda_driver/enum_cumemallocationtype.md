# enum CUmemAllocationType

Defines the allocation types available

###### Values

**CU_MEM_ALLOCATION_TYPE_INVALID = 0x0**
**CU_MEM_ALLOCATION_TYPE_PINNED = 0x1**
This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is
actively using it
**CU_MEM_ALLOCATION_TYPE_MANAGED = 0x2**
This allocation type is managed memory
**CU_MEM_ALLOCATION_TYPE_MAX = 0x7FFFFFFF**


CUDA Driver API TRM-06703-001 _vRelease Version  |  64


Modules