# 10.18.6. Restrictions

### 10.18.6. Restrictions[](#restrictions "Permalink to this headline")

`__assume()` is only supported when using `cl.exe` host compiler. The other functions are supported on all platforms, subject to the following restrictions:

* If the host compiler supports the function, the function can be invoked from anywhere in translation unit.
* Otherwise, the function must be invoked from within the body of a `__device__`/ `__global__`function, or only when the `__CUDA_ARCH__` macro is defined[5](#fn12).