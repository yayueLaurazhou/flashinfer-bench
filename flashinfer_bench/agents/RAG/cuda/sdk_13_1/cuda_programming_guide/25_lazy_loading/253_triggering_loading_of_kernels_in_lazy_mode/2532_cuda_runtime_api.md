# 25.3.2. CUDA Runtime API

### 25.3.2. CUDA Runtime API[](#cuda-runtime-api "Permalink to this headline")

CUDA Runtime API manages module management automatically,
so we recommend simply using `cudaFuncGetAttributes()` to reference the kernel.

This will ensure that the kernel is loaded without changing the state.