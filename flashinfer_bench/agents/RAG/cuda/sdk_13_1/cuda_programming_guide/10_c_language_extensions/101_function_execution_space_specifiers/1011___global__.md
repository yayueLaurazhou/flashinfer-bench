# 10.1.1. __global__

### 10.1.1. \_\_global\_\_[](#global "Permalink to this headline")

The `__global__` execution space specifier declares a function as being a kernel. Such a function is:

* Executed on the device,
* Callable from the host,
* Callable from the device for devices of compute capability 5.0 or higher (see [CUDA Dynamic Parallelism](#cuda-dynamic-parallelism) for more details).

A `__global__` function must have void return type, and cannot be a member of a class.

Any call to a `__global__` function must specify its execution configuration as described in [Execution Configuration](#execution-configuration).

A call to a `__global__` function is asynchronous, meaning it returns before the device has completed its execution.