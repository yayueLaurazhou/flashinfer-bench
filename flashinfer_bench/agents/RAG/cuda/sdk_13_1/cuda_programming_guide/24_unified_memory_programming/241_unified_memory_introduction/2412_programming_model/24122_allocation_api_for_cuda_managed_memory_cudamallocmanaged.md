# 24.1.2.2. Allocation API for CUDA Managed Memory: cudaMallocManaged()

#### 24.1.2.2. Allocation API for CUDA Managed Memory: `cudaMallocManaged()`[](#allocation-api-for-cuda-managed-memory-cudamallocmanaged "Permalink to this headline")

On systems with CUDA Managed Memory support, unified memory may be allocated using:

```
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size);
```

This API is syntactically identical to `cudaMalloc()`: it allocates `size` bytes of managed memory and
sets `devPtr` to refer to the allocation.
CUDA Managed Memory is also deallocated with `cudaFree()`.

On [systems with full CUDA Managed Memory support](#um-requirements), managed memory allocations
may be accessed concurrently by all CPUs and GPUs in the system.
Replacing host calls to `cudaMalloc()` with `cudaMallocManaged()` does not impact program semantics on these systems;
device code is not able to call `cudaMallocManaged()`.

The following example shows the use of `cudaMallocManaged()`:

```
__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s;
  cudaMallocManaged(&s, 100);
  // Note direct Host-code use of "s"
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}
```

Note

For systems that support CUDA Managed Memory allocations, but do not provide full support,
see [Coherency and Concurrency](#um-coherency-hd).
Implementation details (may change any time):

* Devices of compute capability 5.x allocate CUDA Managed Memory on the GPU.
* Devices of compute capability 6.x and greater populate the memory on first touch, just like System-Allocated Memory APIs.