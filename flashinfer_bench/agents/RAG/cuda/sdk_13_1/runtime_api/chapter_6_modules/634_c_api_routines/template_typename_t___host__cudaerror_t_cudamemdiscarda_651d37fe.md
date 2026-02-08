# template < typename T > __host__cudaError_t cudaMemDiscardAndPrefetchBatchAsync (T **dptrs, size_t *sizes, size_t count, cudaMemLocation prefetchLocs, unsigned long long flags, cudaStream_t stream)

Performs a batch of memory discard and prefetches asynchronously.

##### Description

This is an alternate spelling for cudaMemDiscardAndPrefetchBatchAsync made available through
function overloading.

The cudaMemLocation specified by prefetchLocs are applicable for all the operations in the batch.

See also:

cudaMemDiscardAndPrefetchBatchAsync


CUDA Runtime API vRelease Version  |  499


Modules