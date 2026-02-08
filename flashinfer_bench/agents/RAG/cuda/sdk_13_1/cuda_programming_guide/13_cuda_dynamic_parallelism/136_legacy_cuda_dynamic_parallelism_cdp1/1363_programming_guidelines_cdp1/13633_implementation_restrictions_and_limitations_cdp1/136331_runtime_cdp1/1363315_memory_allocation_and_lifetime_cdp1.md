# 13.6.3.3.1.5. Memory Allocation and Lifetime (CDP1)

###### 13.6.3.3.1.5. Memory Allocation and Lifetime (CDP1)[](#memory-allocation-and-lifetime-cdp1 "Permalink to this headline")

See [Memory Allocation and Lifetime](#memory-allocation-and-lifetime), above, for CDP2 version of document.

`cudaMalloc()` and `cudaFree()` have distinct semantics between the host and device environments. When invoked from the host, `cudaMalloc()` allocates a new region from unused device memory. When invoked from the device runtime these functions map to device-side `malloc()` and `free()`. This implies that within the device environment the total allocatable memory is limited to the device `malloc()` heap size, which may be smaller than the available unused device memory. Also, it is an error to invoke `cudaFree()` from the host program on a pointer which was allocated by `cudaMalloc()` on the device or vice-versa.

|  | `cudaMalloc()` on Host | `cudaMalloc()` on Device |
| --- | --- | --- |
| `cudaFree()` on Host | Supported | Not Supported |
| `cudaFree()` on Device | Not Supported | Supported |
| Allocation limit | Free device memory | `cudaLimitMallocHeapSize` |