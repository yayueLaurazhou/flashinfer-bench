# 10.2. Variable Memory Space Specifiers

## 10.2. Variable Memory Space Specifiers[](#variable-memory-space-specifiers "Permalink to this headline")

Variable memory space specifiers denote the memory location on the device of a variable.

An automatic variable declared in device code without any of the `__device__`, `__shared__` and `__constant__` memory space specifiers described in this section generally resides in a register. However in some cases the compiler might choose to place it in local memory, which can have adverse performance consequences as detailed in [Device Memory Accesses](#device-memory-accesses).