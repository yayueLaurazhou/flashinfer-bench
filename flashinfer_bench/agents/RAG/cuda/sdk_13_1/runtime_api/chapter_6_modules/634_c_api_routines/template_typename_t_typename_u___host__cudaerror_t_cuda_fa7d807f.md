# template < typename T, typename U > __host__cudaError_t cudaMemcpyBatchAsync (const T **dsts, const U **srcs, const size_t *sizes, size_t count, cudaMemcpyAttributes attr, cudaStream_t hStream)

Performs a batch of memory copies asynchronously.

##### Description

This is an alternate spelling for cudaMemcpyBatchAsync made available through function overloading.

The cudaMemcpyAttributes specified by attr are applicable for all the copies specified in the batch.

See also:

cudaMemcpyBatchAsync


CUDA Runtime API vRelease Version  |  493


Modules