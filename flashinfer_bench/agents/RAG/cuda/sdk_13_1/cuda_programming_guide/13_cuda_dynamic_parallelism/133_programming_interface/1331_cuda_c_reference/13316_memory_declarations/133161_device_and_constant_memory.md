# 13.3.1.6.1. Device and Constant Memory

##### 13.3.1.6.1. Device and Constant Memory[](#device-and-constant-memory "Permalink to this headline")

Memory declared at file scope with `__device__` or `__constant__` memory space specifiers behaves identically when using the device runtime. All kernels may read or write device variables, whether the kernel was initially launched by the host or device runtime. Equivalently, all kernels will have the same view of `__constant__`s as declared at the module scope.