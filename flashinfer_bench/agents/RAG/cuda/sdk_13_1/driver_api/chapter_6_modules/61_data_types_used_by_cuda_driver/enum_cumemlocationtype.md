# enum CUmemLocationType

Specifies the type of location

###### Values

**CU_MEM_LOCATION_TYPE_INVALID = 0x0**
**CU_MEM_LOCATION_TYPE_NONE = 0x0**
Location is unspecified. This is used when creating a managed memory pool to indicate no preferred
location for the pool
**CU_MEM_LOCATION_TYPE_DEVICE = 0x1**
Location is a device location, thus id is a device ordinal
**CU_MEM_LOCATION_TYPE_HOST = 0x2**
Location is host, id is ignored
**CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3**
Location is a host NUMA node, thus id is a host NUMA node id
**CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4**
Location is a host NUMA node of the current thread, id is ignored
**CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF**


CUDA Driver API TRM-06703-001 _vRelease Version  |  66


Modules