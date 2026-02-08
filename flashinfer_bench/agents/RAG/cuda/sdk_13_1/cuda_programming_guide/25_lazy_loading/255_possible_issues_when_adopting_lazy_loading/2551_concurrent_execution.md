# 25.5.1. Concurrent Execution

### 25.5.1. Concurrent Execution[](#concurrent-execution "Permalink to this headline")

Loading kernels might require context synchronization.
Some programs incorrectly treat the possibility of concurrent execution of kernels as a guarantee.
In such cases, if program assumes that two kernels will be able to execute concurrently,
and one of the kernels will not return without the other kernel executing, there is a possibility of a deadlock.

If kernel A will be spinning in an infinite loop until kernel B is executing.
In such case launching kernel B will trigger lazy loading of kernel B. If this loading will require context synchronization,
then we have a deadlock: kernel A is waiting for kernel B, but loading kernel B is stuck waiting for kernel A to finish to synchronize the context.

Such program is an anti-pattern, but if for any reason you want to keep it you can do the following:

* preload all kernels that you hope to execute concurrently prior to launching them
* run application with `CUDA_MODULE_DATA_LOADING=EAGER` to force loading data eagerly without forcing each function to load eagerly