# 24.1.2.3. Global-Scope Managed Variables Using __managed__

#### 24.1.2.3. Global-Scope Managed Variables Using `__managed__`[](#global-scope-managed-variables-using-managed "Permalink to this headline")

CUDA `__managed__` variables behave as if they were allocated via `cudaMallocManaged()`
(see [Allocation API for CUDA Managed Memory: cudaMallocManaged()](#um-explicit-allocation)).
They simplify programs with global variables, making it particularly easy to exchange data between host and device
without manual allocations or copying.

On [systems with full CUDA Unified Memory support](#um-requirements),
file-scope or global-scope variables cannot be directly accessed by device code.
But a pointer to these variables may be passed to the kernel as an argument,
see [System-Allocated Memory: in-depth examples](#um-system-allocator) for examples.

System Allocator

```
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

int main() {
  // Requires System-Allocated Memory support
  int value;
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```


Managed

```
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

// Requires CUDA Managed Memory support
__managed__ int value;

int main() {
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```

Note the absence of explicit `cudaMemcpy()` commands and the fact that the written value `value` is visible on both CPU and GPU.

CUDA `__managed__` variable implies `__device__` and is equivalent to `__managed__ __device__`, which is also allowed.
Variables marked `__constant__` may not be marked as `__managed__`.

A valid CUDA context is necessary for the correct operation of `__managed__` variables.
Accessing `__managed__` variables can trigger CUDA context creation if a context for the current device hasn’t already been created.
In the example above, accessing `value` before the kernel launch triggers context creation on the default device.
In the absence of that access, the kernel launch would have triggered context creation.

C++ objects declared as `__managed__` are subject to certain specific constraints, particularly where static initializers are concerned.
Please refer to [C++ Language Support](#c-cplusplus-language-support) for a list of these constraints.

Note

For [devices with CUDA Managed Memory without full support](#um-requirements),
visibility of `__managed__` variables for asynchronous operations executing in CUDA streams
is discussed in the section on
[Managing Data Visibility and Concurrent CPU + GPU Access with Streams](#um-managing-data).