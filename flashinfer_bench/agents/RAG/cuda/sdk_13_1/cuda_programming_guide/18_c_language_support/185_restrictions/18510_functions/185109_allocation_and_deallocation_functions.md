# 18.5.10.9. Allocation and Deallocation Functions

#### 18.5.10.9. Allocation and Deallocation Functions[](#allocation-and-deallocation-functions "Permalink to this headline")

A user-defined `operator new`, `operator new[]`, `operator delete`, or `operator delete[]` cannot be used to replace the corresponding `__host__` or `__device__` builtins provided by the compiler.