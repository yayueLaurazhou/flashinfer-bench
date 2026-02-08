# __host__cudaError_t cudaMemDiscardBatchAsync (void **dptrs, size_t *sizes, size_t count, unsigned long long flags, cudaStream_t stream)

Performs a batch of memory discards asynchronously.

##### Parameters

**dptrs**

  - Array of pointers to be discarded
**sizes**

  - Array of sizes for memory discard operations.
**count**

  - Size of dptrs and sizes arrays.
**flags**

  - Flags reserved for future use. Must be zero.
**stream**

##### Description

Performs a batch of memory discards. The batch as a whole executes in stream order but operations
within a batch are not guaranteed to execute in any specific order. All devices in the system must have
a non-zero value for the device attribute cudaDevAttrConcurrentManagedAccess otherwise the API
will return an error.

Discarding a memory range informs the driver that the contents of that range are no longer useful.
Discarding memory ranges allows the driver to optimize certain data migrations and can also help
reduce memory pressure. This operation can be undone on any part of the range by either writing
to it or prefetching it via cudaMemPrefetchAsync or cudaMemPrefetchBatchAsync. Reading from
a discarded range, without a subsequent write or prefetch to that part of the range, will return an
indeterminate value. Note that any reads, writes or prefetches to any part of the memory range that
occur simultaneously with the discard operation result in undefined behavior.

Performs memory discard on address ranges specified in dptrs and sizes. Both arrays
must be of the same length as specified by count. Each memory range specified must refer to
managed memory allocated via cudaMallocManaged or declared via __managed__ variables
or it may also refer to system-allocated memory when all devices have a non-zero value for
cudaDevAttrPageableMemoryAccess.


CUDA Runtime API vRelease Version  |  184


Modules