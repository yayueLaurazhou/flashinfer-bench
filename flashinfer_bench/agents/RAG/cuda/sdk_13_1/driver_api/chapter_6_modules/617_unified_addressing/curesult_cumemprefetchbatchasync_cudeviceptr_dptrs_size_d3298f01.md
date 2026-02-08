# CUresult cuMemPrefetchBatchAsync (CUdeviceptr *dptrs, size_t *sizes, size_t count, CUmemLocation *prefetchLocs, size_t *prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream)

Performs a batch of memory prefetches asynchronously.

###### Parameters

**dptrs**

  - Array of pointers to be prefetched
**sizes**

  - Array of sizes for memory prefetch operations.
**count**

  - Size of dptrs and sizes arrays.
**prefetchLocs**

  - Array of locations to prefetch to.
**prefetchLocIdxs**

  - Array of indices to specify which operands each entry in the prefetchLocs array
applies to. The locations specified in prefetchLocs[k] will be applied to copies starting from
prefetchLocIdxs[k] through prefetchLocIdxs[k+1] - 1. Also prefetchLocs[numPrefetchLocs - 1] will
apply to prefetches starting from prefetchLocIdxs[numPrefetchLocs - 1] through count - 1.
**numPrefetchLocs**

  - Size of prefetchLocs and prefetchLocIdxs arrays.
**flags**

  - Flags reserved for future use. Must be zero.


CUDA Driver API TRM-06703-001 _vRelease Version  |  320


Modules


**hStream**

  - The stream to enqueue the operations in. Must not be legacy NULL stream.

###### Description

Performs a batch of memory prefetches. The batch as a whole executes in stream
order but operations within a batch are not guaranteed to execute in any specific
order. All devices in the system must have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS otherwise the API will return
an error.

The semantics of the individual prefetch operations are as described in cuMemPrefetchAsync.

Performs memory prefetch on address ranges specified in dptrs and sizes. Both arrays
must be of the same length as specified by count. Each memory range specified must refer to
managed memory allocated via cuMemAllocManaged or declared via __managed__ variables
or it may also refer to system-allocated memory when all devices have a non-zero value for
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. The prefetch location for every
operation in the batch is specified in the prefetchLocs array. Each entry in this array can apply
to more than one operation. This can be done by specifying in the prefetchLocIdxs array, the
index of the first prefetch operation that the corresponding entry in the prefetchLocs array applies
to. Both prefetchLocs and prefetchLocIdxs must be of the same length as specified by
numPrefetchLocs. For example, if a batch has 10 prefetches listed in dptrs/sizes, the first 4 of
which are to be prefetched to one location and the remaining 6 are to be prefetched to another, then
numPrefetchLocs will be 2, prefetchLocIdxs will be {0, 4} and prefetchLocs will
contain the two locations. Note the first entry in prefetchLocIdxs must always be 0. Also,
each entry must be greater than the previous entry and the last entry should be less than count.
Furthermore, numPrefetchLocs must be lesser than or equal to count.