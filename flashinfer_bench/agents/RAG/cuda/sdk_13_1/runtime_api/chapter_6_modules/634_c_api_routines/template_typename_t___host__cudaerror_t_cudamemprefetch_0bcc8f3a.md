# template < typename T > __host__cudaError_t cudaMemPrefetchBatchAsync (T **dptrs, size_t *sizes, size_t count, cudaMemLocation *prefetchLocs, size_t *prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream)

Performs a batch of memory prefetches asynchronously.

##### Description

This is an alternate spelling for cudaMemPrefetchBatchAsync made available through function
overloading.

See also:

cudaMemPrefetchBatchAsync