# 10.2.2. __constant__

### 10.2.2. \_\_constant\_\_[](#constant "Permalink to this headline")

The `__constant__` memory space specifier, optionally used together with `__device__`, declares a variable that:

* Resides in constant memory space,
* Has the lifetime of the CUDA context in which it is created,
* Has a distinct object per device,
* Is accessible from all the threads within the grid and from the host through the runtime library (`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).

The behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point of this grid’s lifetime is undefined.