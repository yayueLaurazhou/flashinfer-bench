# __host__cudaError_t cudaMemAdvise (const void *devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location)

Advise about the usage of a given memory range.

##### Parameters

**devPtr**

  - Pointer to memory to set the advice for
**count**

  - Size in bytes of the memory range
**advice**

  - Advice to be applied for the specified memory range


CUDA Runtime API vRelease Version  |  147


Modules


**location**

  - location to apply the advice for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Advise the Unified Memory subsystem about the usage pattern for the memory range starting at
devPtr with a size of count bytes. The start address and end address of the memory range will be
rounded down and rounded up respectively to be aligned to CPU page size before the advice is applied.
The memory range must refer to managed memory allocated via cudaMallocManaged or declared via
__managed__ variables. The memory range could also refer to system-allocated pageable memory
provided it represents a valid, host-accessible region of memory and all additional constraints imposed
by advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
memory range results in an error being returned.

The advice parameter can take the following values:

cudaMemAdviseSetReadMostly: This implies that the data is mostly going to be read from

##### **‣**

and only occasionally written to. Any read accesses from any processor to this region will
create a read-only copy of at least the accessed pages in that processor's memory. Additionally,
if cudaMemPrefetchAsync or cudaMemPrefetchAsync is called on this region, it will
create a read-only copy of the data on the destination processor. If the target location for
cudaMemPrefetchAsync is a host NUMA node and a read-only copy already exists on another
host NUMA node, that copy will be migrated to the targeted host NUMA node. If any processor
writes to this region, all copies of the corresponding page will be invalidated except for the one
where the write occurred. If the writing processor is the CPU and the preferred location of the
page is a host NUMA node, then the page will also be migrated to that host NUMA node. The
location argument is ignored for this advice. Note that for a page to be read-duplicated, the
accessing processor must either be the CPU or a GPU that has a non-zero value for the device
attribute cudaDevAttrConcurrentManagedAccess. Also, if a context is created on a device that does
not have the device attribute cudaDevAttrConcurrentManagedAccess set, then read-duplication
will not occur until all such contexts are destroyed. If the memory region refers to valid systemallocated pageable memory, then the accessing device must have a non-zero value for the device
attribute cudaDevAttrPageableMemoryAccess for a read-only copy to be created on that device.
Note however that if the accessing device also has a non-zero value for the device attribute
cudaDevAttrPageableMemoryAccessUsesHostPageTables, then setting this advice will not create a
read-only copy when that device accesses this memory region.

cudaMemAdviceUnsetReadMostly: Undoes the effect of cudaMemAdviseSetReadMostly and also

##### **‣**

prevents the Unified Memory driver from attempting heuristic read-duplication on the memory
range. Any read-duplicated copies of the data will be collapsed into a single copy. The location for
the collapsed copy will be the preferred location if the page has a preferred location and one of the


CUDA Runtime API vRelease Version  |  148


Modules


read-duplicated copies was resident at that location. Otherwise, the location chosen is arbitrary.
Note: The location argument is ignored for this advice.

cudaMemAdviseSetPreferredLocation: This advice sets the preferred location for the data to be the

##### **‣**

memory belonging to location. When cudaMemLocation::type is cudaMemLocationTypeHost,
cudaMemLocation::id is ignored and the preferred location is set to be host memory. To set the
preferred location to a specific host NUMA node, applications must set cudaMemLocation::type to
cudaMemLocationTypeHostNuma and cudaMemLocation::id must specify the NUMA ID of the
host NUMA node. If cudaMemLocation::type is set to cudaMemLocationTypeHostNumaCurrent,
cudaMemLocation::id will be ignored and the host NUMA node closest to the calling
thread's CPU will be used as the preferred location. If cudaMemLocation::type is a
cudaMemLocationTypeDevice, then cudaMemLocation::id must be a valid device ordinal and the
device must have a non-zero value for the device attribute cudaDevAttrConcurrentManagedAccess.
Setting the preferred location does not cause data to migrate to that location immediately. Instead,
it guides the migration policy when a fault occurs on that memory region. If the data is already
in its preferred location and the faulting processor can establish a mapping without requiring
the data to be migrated, then data migration will be avoided. On the other hand, if the data is
not in its preferred location or if a direct mapping cannot be established, then it will be migrated
to the processor accessing it. It is important to note that setting the preferred location does not
prevent data prefetching done using cudaMemPrefetchAsync. Having a preferred location can
override the page thrash detection and resolution logic in the Unified Memory driver. Normally,
if a page is detected to be constantly thrashing between for example host and device memory,
the page may eventually be pinned to host memory by the Unified Memory driver. But if the
preferred location is set as device memory, then the page will continue to thrash indefinitely. If
cudaMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
policies associated with that advice will override the policies of this advice, unless read accesses
from location will not result in a read-only copy being created on that procesor as outlined in
description for the advice cudaMemAdviseSetReadMostly. If the memory region refers to valid
system-allocated pageable memory, and cudaMemLocation::type is cudaMemLocationTypeDevice
then cudaMemLocation::id must be a valid device that has a non-zero alue for the device attribute
cudaDevAttrPageableMemoryAccess.

cudaMemAdviseUnsetPreferredLocation: Undoes the effect of

##### **‣**

cudaMemAdviseSetPreferredLocation and changes the preferred location to none. The location
argument is ignored for this advice.

cudaMemAdviseSetAccessedBy: This advice implies that the data will be accessed by processor

##### **‣**

location. The cudaMemLocation::type must be either cudaMemLocationTypeDevice with
cudaMemLocation::id representing a valid device ordinal or cudaMemLocationTypeHost and
cudaMemLocation::id will be ignored. All other location types are invalid. If cudaMemLocation::id
is a GPU, then the device attribute cudaDevAttrConcurrentManagedAccess must be non-zero.
This advice does not cause data migration and has no impact on the location of the data per se.
Instead, it causes the data to always be mapped in the specified processor's page tables, as long
as the location of the data permits a mapping to be established. If the data gets migrated for
any reason, the mappings are updated accordingly. This advice is recommended in scenarios


CUDA Runtime API vRelease Version  |  149


Modules


where data locality is not important, but avoiding faults is. Consider for example a system
containing multiple GPUs with peer-to-peer access enabled, where the data located on one
GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data over to the other
GPUs is not as important because the accesses are infrequent and the overhead of migration
may be too high. But preventing faults can still help improve performance, and so having a
mapping set up in advance is useful. Note that on CPU access of this data, the data may be
migrated to host memory because the CPU typically cannot access device memory directly.
Any GPU that had the cudaMemAdviseSetAccessedBy flag set for this data will now have
its mapping updated to point to the page in host memory. If cudaMemAdviseSetReadMostly
is also set on this memory region or any subset of it, then the policies associated with that
advice will override the policies of this advice. Additionally, if the preferred location of
this memory region or any subset of it is also location, then the policies associated
with CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of
this advice. If the memory region refers to valid system-allocated pageable memory, and
cudaMemLocation::type is cudaMemLocationTypeDevice then device in cudaMemLocation::id
must have a non-zero value for the device attribute cudaDevAttrPageableMemoryAccess.
Additionally, if cudaMemLocation::id has a non-zero value for the device attribute
cudaDevAttrPageableMemoryAccessUsesHostPageTables, then this call has no effect.

CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of

##### **‣**

cudaMemAdviseSetAccessedBy. Any mappings to the data from location may be removed at
any time causing accesses to result in non-fatal page faults. If the memory region refers to valid
system-allocated pageable memory, and cudaMemLocation::type is cudaMemLocationTypeDevice
then device in cudaMemLocation::id must have a non-zero value for the device attribute
cudaDevAttrPageableMemoryAccess. Additionally, if cudaMemLocation::id has a non-zero value
for the device attribute cudaDevAttrPageableMemoryAccessUsesHostPageTables, then this call
has no effect.













See also:

cudaMemcpy, cudaMemcpyPeer, cudaMemcpyAsync, cudaMemcpy3DPeerAsync,
cudaMemPrefetchAsync, cuMemAdvise


CUDA Runtime API vRelease Version  |  150


Modules