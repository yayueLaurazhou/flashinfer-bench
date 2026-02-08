# CUresult cuMemPrefetchAsync (CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream)

Prefetches memory to the specified destination location.

###### Parameters

**devPtr**

  - Pointer to be prefetched
**count**

  - Size in bytes
**location**

  - Location to prefetch to
**flags**

  - flags for future use, must be zero now.
**hStream**

  - Stream to enqueue prefetch operation

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE


CUDA Driver API TRM-06703-001 _vRelease Version  |  318


Modules

###### Description

Prefetches memory to the specified destination location. devPtr is the base device pointer of the
memory to be prefetched and location specifies the destination location. count specifies the
number of bytes to copy. hStream is the stream in which the operation is enqueued. The memory
range must refer to managed memory allocated via cuMemAllocManaged, via cuMemAllocFromPool
from a managed memory pool or declared via __managed__ variables.

Specifying CU_MEM_LOCATION_TYPE_DEVICE for CUmemLocation::type will prefetch
memory to GPU specified by device ordinal CUmemLocation::id which must have non-zero value
for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
Additionally, hStream must be associated with a device that has a non-zero value for the
device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
Specifying CU_MEM_LOCATION_TYPE_HOST as CUmemLocation::type will prefetch
data to host memory. Applications can request prefetching memory to a specific host NUMA
node by specifying CU_MEM_LOCATION_TYPE_HOST_NUMA for CUmemLocation::type
and a valid host NUMA node id in CUmemLocation::id Users can also request prefetching
memory to the host NUMA node closest to the current thread's CPU by specifying
CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT for CUmemLocation::type.
Note when CUmemLocation::type is etiher CU_MEM_LOCATION_TYPE_HOST OR
CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT, CUmemLocation::id will be ignored.

The start address and end address of the memory range will be rounded down and rounded up
respectively to be aligned to CPU page size before the prefetch operation is enqueued in the stream.

If no physical memory has been allocated for this region, then this memory region will be populated
and mapped on the destination device. If there's insufficient memory to prefetch the desired region, the
Unified Memory driver may evict pages from other cuMemAllocManaged allocations to host memory
in order to make room. Device memory allocated using cuMemAlloc or cuArrayCreate will not be
evicted.

By default, any mappings to the previous location of the migrated pages are removed and mappings for
the new location are only setup on the destination location. The exact behavior however also depends
on the settings applied to this memory range via cuMemAdvise as described below:

If CU_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range, then
that subset will create a read-only copy of the pages on destination location. If however the destination
location is a host NUMA node, then any pages of that subset that are already in another host NUMA
node will be transferred to the destination.

If CU_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory
range, then the pages will be migrated to location even if location is not the preferred location
of any pages in the memory range.

If CU_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range, then
mappings to those pages from all the appropriate processors are updated to refer to the new location if
establishing such a mapping is possible. Otherwise, those mappings are cleared.


CUDA Driver API TRM-06703-001 _vRelease Version  |  319


Modules


Note that this API is not required for functionality and only serves to improve performance by allowing
the application to migrate data to a suitable location before it is accessed. Memory accesses to this
range are always coherent and are allowed even when the data is actively being migrated.

Note that this function is asynchronous with respect to the host and all work on other devices.











See also:

cuMemcpy, cuMemcpyPeer, cuMemcpyAsync, cuMemcpy3DPeerAsync, cuMemAdvise,
cudaMemPrefetchAsync