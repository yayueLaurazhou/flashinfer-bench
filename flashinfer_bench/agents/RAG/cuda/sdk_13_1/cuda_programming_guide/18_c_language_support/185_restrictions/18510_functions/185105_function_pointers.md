# 18.5.10.5. Function Pointers

#### 18.5.10.5. Function Pointers[](#function-pointers "Permalink to this headline")

The address of a `__global__` function taken in host code cannot be used in device code (e.g. to launch the kernel). Similarly, the address of a `__global__` function taken in device code cannot be used in host code.

It is not allowed to take the address of a `__device__` function in host code.