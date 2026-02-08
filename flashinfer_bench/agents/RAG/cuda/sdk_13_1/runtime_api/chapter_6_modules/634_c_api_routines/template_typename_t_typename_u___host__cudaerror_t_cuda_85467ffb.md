# template < typename T, typename U > __host__cudaError_t cudaMemcpyBatchAsync (const T **dsts, const U **srcs, const size_t *sizes, size_t count, cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, cudaStream_t hStream)

Performs a batch of memory copies asynchronously.

##### Description

This is an alternate spelling for cudaMemcpyBatchAsync made available through function overloading.

See also:

cudaMemcpyBatchAsync