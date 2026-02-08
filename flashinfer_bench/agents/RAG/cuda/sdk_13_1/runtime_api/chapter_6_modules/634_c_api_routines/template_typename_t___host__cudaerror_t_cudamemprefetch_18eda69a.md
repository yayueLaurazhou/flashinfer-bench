# template < typename T > __host__cudaError_t cudaMemPrefetchBatchAsync (T **dptrs, size_t *sizes, size_t count, cudaMemLocation prefetchLocs, unsigned long long flags, cudaStream_t stream)

Performs a batch of memory prefetches asynchronously.

##### Description

This is an alternate spelling for cudaMemPrefetchBatchAsync made available through function
overloading.

The cudaMemLocation specified by prefetchLocs are applicable for all the prefetches specified in
the batch.

See also:

cudaMemPrefetchBatchAsync


CUDA Runtime API vRelease Version  |  500


Modules