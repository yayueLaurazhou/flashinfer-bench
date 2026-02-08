# 10.1.4. Undefined behavior

### 10.1.4. Undefined behavior[](#undefined-behavior "Permalink to this headline")

A ‘cross-execution space’ call has undefined behavior when:

* `__CUDA_ARCH__` is defined, a call from within a `__global__`, `__device__` or `__host__ __device__` function to a `__host__` function.
* `__CUDA_ARCH__` is undefined, a call from within a `__host__` function to a `__device__` function. [4](#fn11)