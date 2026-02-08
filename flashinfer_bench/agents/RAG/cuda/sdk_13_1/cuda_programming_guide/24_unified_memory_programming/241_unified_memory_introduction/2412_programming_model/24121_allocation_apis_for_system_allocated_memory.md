# 24.1.2.1. Allocation APIs for System-Allocated Memory

#### 24.1.2.1. Allocation APIs for System-Allocated Memory[](#allocation-apis-for-system-allocated-memory "Permalink to this headline")

On [systems with full CUDA Unified Memory support](#um-requirements), all memory is unified memory.
This includes memory allocated with system allocation APIs, such as `malloc()`, `mmap()`, C++ `new()` operator,
and also automatic variables on CPU thread stacks, thread locals, global variables, and so on.

System-Allocated Memory may be populated on first touch, depending on the API and system settings used.
First touch means that:

* The allocation APIs allocate virtual memory and return immediately, and
* physical memory is populated when a thread accesses the memory for the first time.

Usually, the physical memory will be chosen “close” to the processor that thread is running on. For example,

* GPU thread accesses it first: physical GPU memory of GPU that thread runs on is chosen.
* CPU thread accesses it first: physical CPU memory in the memory NUMA node of the CPU core that thread runs on is chosen.

CUDA Unified Memory Hint and Prefetch APIs, `cudaMemAdvise` and `cudaMemPreftchAsync`, may be used on System-Allocated Memory.
These APIs are covered below in the [Data Usage Hints](#um-tuning-usage) section.

```
__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s = (char*)malloc(100);
  // Physical allocation placed in CPU memory because host accesses "s" first
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}
```