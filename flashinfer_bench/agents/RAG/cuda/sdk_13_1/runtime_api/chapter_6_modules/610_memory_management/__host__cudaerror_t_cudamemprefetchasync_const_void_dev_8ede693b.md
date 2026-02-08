# __host__cudaError_t cudaMemPrefetchAsync (const void *devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream)

Prefetches memory to the specified destination location.

##### Parameters

**devPtr**

  - Pointer to be prefetched
**count**

  - Size in bytes
**location**

  - location to prefetch to
**flags**

  - flags for future use, must be zero now.
**stream**

  - Stream to enqueue prefetch operation

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Prefetches memory to the specified destination location. devPtr is the base device pointer of the
memory to be prefetched and location specifies the destination location. count specifies the
number of bytes to copy. stream is the stream in which the operation is enqueued. The memory
range must refer to managed memory allocated via cudaMallocManaged or declared via __managed__
variables, or it may also refer to memory allocated from a managed memory pool, or it may also refer
to system-allocated memory on systems with non-zero cudaDevAttrPageableMemoryAccess.

Specifying cudaMemLocationTypeDevice for cudaMemLocation::type will prefetch memory to
GPU specified by device ordinal cudaMemLocation::id which must have non-zero value for the
device attribute concurrentManagedAccess. Additionally, stream must be associated with a
device that has a non-zero value for the device attribute concurrentManagedAccess. Specifying
cudaMemLocationTypeHost as cudaMemLocation::type will prefetch data to host memory.
Applications can request prefetching memory to a specific host NUMA node by specifying
cudaMemLocationTypeHostNuma for cudaMemLocation::type and a valid host NUMA node
id in cudaMemLocation::id Users can also request prefetching memory to the host NUMA node
closest to the current thread's CPU by specifying cudaMemLocationTypeHostNumaCurrent for
cudaMemLocation::type. Note when cudaMemLocation::type is etiher cudaMemLocationTypeHost
OR cudaMemLocationTypeHostNumaCurrent, cudaMemLocation::id will be ignored.


CUDA Runtime API vRelease Version  |  186


Modules


The start address and end address of the memory range will be rounded down and rounded up
respectively to be aligned to CPU page size before the prefetch operation is enqueued in the stream.

If no physical memory has been allocated for this region, then this memory region will be populated
and mapped on the destination device. If there's insufficient memory to prefetch the desired region, the
Unified Memory driver may evict pages from other cudaMallocManaged allocations to host memory
in order to make room. Device memory allocated using cudaMalloc or cudaMallocArray will not be
evicted.

By default, any mappings to the previous location of the migrated pages are removed and mappings for
the new location are only setup on the destination location. The exact behavior however also depends
on the settings applied to this memory range via cuMemAdvise as described below:

If cudaMemAdviseSetReadMostly was set on any subset of this memory range, then that subset will
create a read-only copy of the pages on destination location. If however the destination location is a
host NUMA node, then any pages of that subset that are already in another host NUMA node will be
transferred to the destination.

If cudaMemAdviseSetPreferredLocation was called on any subset of this memory range, then the pages
will be migrated to location even if location is not the preferred location of any pages in the
memory range.

If cudaMemAdviseSetAccessedBy was called on any subset of this memory range, then mappings to
those pages from all the appropriate processors are updated to refer to the new location if establishing
such a mapping is possible. Otherwise, those mappings are cleared.

Note that this API is not required for functionality and only serves to improve performance by allowing
the application to migrate data to a suitable location before it is accessed. Memory accesses to this
range are always coherent and are allowed even when the data is actively being migrated.

Note that this function is asynchronous with respect to the host and all work on other devices.













See also:

cudaMemcpy, cudaMemcpyPeer, cudaMemcpyAsync, cudaMemcpy3DPeerAsync, cudaMemAdvise,
cuMemPrefetchAsync


CUDA Runtime API vRelease Version  |  187


Modules