# __host__cudaError_t cudaMemDiscardAndPrefetchBatchAsync (void **dptrs, size_t *sizes, size_t count, cudaMemLocation *prefetchLocs, size_t *prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream)

Performs a batch of memory discards and prefetches asynchronously.

##### Parameters

**dptrs**

  - Array of pointers to be discarded
**sizes**

  - Array of sizes for memory discard operations.
**count**

  - Size of dptrs and sizes arrays.
**prefetchLocs**

  - Array of locations to prefetch to.
**prefetchLocIdxs**

  - Array of indices to specify which operands each entry in the prefetchLocs array applies
to. The locations specified in prefetchLocs[k] will be applied to operations starting from
prefetchLocIdxs[k] through prefetchLocIdxs[k+1] - 1. Also prefetchLocs[numPrefetchLocs - 1] will
apply to copies starting from prefetchLocIdxs[numPrefetchLocs - 1] through count - 1.


CUDA Runtime API vRelease Version  |  182


Modules


**numPrefetchLocs**

  - Size of prefetchLocs and prefetchLocIdxs arrays.
**flags**

  - Flags reserved for future use. Must be zero.
**stream**

##### Description

Performs a batch of memory discards followed by prefetches. The batch as a whole executes in stream
order but operations within a batch are not guaranteed to execute in any specific order. All devices in
the system must have a non-zero value for the device attribute cudaDevAttrConcurrentManagedAccess
otherwise the API will return an error.

Calling cudaMemDiscardAndPrefetchBatchAsync is semantically equivalent to calling
cudaMemDiscardBatchAsync followed by cudaMemPrefetchBatchAsync, but is more optimal. For
more details on what discarding and prefetching imply, please refer to cudaMemDiscardBatchAsync
and cudaMemPrefetchBatchAsync respectively. Note that any reads, writes or prefetches to any part of
the memory range that occur simultaneously with this combined discard+prefetch operation result in
undefined behavior.

Performs memory discard and prefetch on address ranges specified in dptrs and sizes. Both
arrays must be of the same length as specified by count. Each memory range specified must refer
to managed memory allocated via cudaMallocManaged or declared via __managed__ variables
or it may also refer to system-allocated memory when all devices have a non-zero value for
cudaDevAttrPageableMemoryAccess. Every operation in the batch has to be associated with a
valid location to prefetch the address range to and specified in the prefetchLocs array. Each
entry in this array can apply to more than one operation. This can be done by specifying in the
prefetchLocIdxs array, the index of the first operation that the corresponding entry in the
prefetchLocs array applies to. Both prefetchLocs and prefetchLocIdxs must be of the
same length as specified by numPrefetchLocs. For example, if a batch has 10 operations listed
in dptrs/sizes, the first 6 of which are to be prefetched to one location and the remaining 4 are to be
prefetched to another, then numPrefetchLocs will be 2, prefetchLocIdxs will be {0, 6} and
prefetchLocs will contain the two set of locations. Note the first entry in prefetchLocIdxs
must always be 0. Also, each entry must be greater than the previous entry and the last entry should be
less than count. Furthermore, numPrefetchLocs must be lesser than or equal to count.


CUDA Runtime API vRelease Version  |  183


Modules