# 10.36.2. Interoperability with Host Memory API

### 10.36.2. Interoperability with Host Memory API[](#interoperability-with-host-memory-api "Permalink to this headline")

Memory allocated via device `malloc()` or `__nv_aligned_device_malloc()` cannot be freed using the runtime (i.e., by calling any of the free memory functions from [Device Memory](#device-memory)).

Similarly, memory allocated via the runtime (i.e., by calling any of the memory allocation functions from [Device Memory](#device-memory)) cannot be freed via `free()`.

In addition, memory allocated by a call to `malloc()` or `__nv_aligned_device_malloc()` in device code cannot be used in any runtime or driver API calls (i.e. cudaMemcpy, cudaMemset, etc).