# 10.2.1. __device__

### 10.2.1. \_\_device\_\_[](#device-variable-specifier "Permalink to this headline")

The `__device__` memory space specifier declares a variable that resides on the device.

At most one of the other memory space specifiers defined in the next three sections may be used together with `__device__` to further denote which memory space the variable belongs to. If none of them is present, the variable:

* Resides in global memory space,
* Has the lifetime of the CUDA context in which it is created,
* Has a distinct object per device,
* Is accessible from all the threads within the grid and from the host through the runtime library `(cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).