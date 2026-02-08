# 25.3.1. CUDA Driver API

### 25.3.1. CUDA Driver API[](#cuda-driver-api "Permalink to this headline")

Loading of kernels happens during `cuModuleGetFunction()` call.
This call is necessary even without Lazy Loading, as it is the only way to obtain a kernel handle.

However, you can also use this API to control with finer granularity when kernels are loaded.