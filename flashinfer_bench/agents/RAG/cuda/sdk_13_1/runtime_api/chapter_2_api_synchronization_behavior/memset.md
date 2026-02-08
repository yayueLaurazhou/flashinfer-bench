# Memset

The cudaMemset functions are asynchronous with respect to the host except when the target memory is
pinned host memory. The Async versions are always asynchronous with respect to the host.