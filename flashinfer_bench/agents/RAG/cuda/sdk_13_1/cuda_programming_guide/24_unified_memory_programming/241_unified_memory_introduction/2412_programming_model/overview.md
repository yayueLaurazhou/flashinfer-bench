# 24.1.2. Programming Model

### 24.1.2. Programming Model[](#um-opt-in "Permalink to this headline")

With CUDA Unified Memory, separate allocations between host and device, and explicit memory transfers between them, are no longer required.
Programs may allocate Unified Memory in the following ways:

* [System-Allocation APIs](#um-implicit-allocation): on [systems with full CUDA Unified Memory support](#um-requirements) via
  any system allocation of the host process (C’s `malloc()`, C++’s `new` operator, POSIX’s `mmap` and so on).
* [CUDA Managed Memory Allocation APIs](#um-explicit-allocation): via the `cudaMallocManaged()` API which is syntactically similar to `cudaMalloc()`.
* [CUDA Managed Variables](#um-language-integration): variables declared with `__managed__`, which are semantically similar to a `__device__` variable.

Most examples in this chapter provide at least two versions, one using CUDA Managed Memory and one using System-Allocated Memory.
Tabs allow you to choose between them. The following samples illustrate how Unified Memory simplifies CUDA programs:

System (`malloc()`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   // Requires System-Allocated Memory support   int* ptr = (int*)malloc(sizeof(int));   write_value<<<1, 1>>>(ptr, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", *ptr);    free(ptr);    return 0; } ``` |


System (Stack)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   // Requires System-Allocated Memory support   int value;   write_value<<<1, 1>>>(&value, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", value);   return 0; } ``` |


Managed (`cudaMallocManaged()`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* ptr = nullptr;   // Requires CUDA Managed Memory support   cudaMallocManaged(&ptr, sizeof(int));   write_value<<<1, 1>>>(ptr, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", *ptr);    cudaFree(ptr);    return 0; } ``` |


Managed (`__managed__`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  // Requires CUDA Managed Memory support __managed__ int value;  int main() {   write_value<<<1, 1>>>(&value, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", value);   return 0; } ``` |

In the example above, the device writes a value which is then read by the host:

* **Without Unified Memory**: both host- and device-side storage for the written value is required (`h_value` and `d_ptr` in the example),
  as is an explicit copy between the two using `cudaMemcpy()`.
* **With Unified Memory**: device accesses data directly from the host. `ptr` / `value` may be used without a separate `h_value` / `d_ptr`
  allocation and no copy routine is required, greatly simplifying and reducing the size of the program. With:

  + **System Allocated**: no other changes required.
  + **Managed Memory**: data allocation changed to use `cudaMallocManaged()`, which returns a pointer valid from both host and device code.