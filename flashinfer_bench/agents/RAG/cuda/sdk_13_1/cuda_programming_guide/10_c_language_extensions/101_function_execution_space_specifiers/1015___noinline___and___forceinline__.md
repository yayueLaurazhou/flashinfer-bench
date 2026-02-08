# 10.1.5. __noinline__ and __forceinline__

### 10.1.5. \_\_noinline\_\_ and \_\_forceinline\_\_[](#noinline-and-forceinline "Permalink to this headline")

The compiler inlines any `__device__` function when deemed appropriate.

The `__noinline__` function qualifier can be used as a hint for the compiler not to inline the function if possible.

The `__forceinline__` function qualifier can be used to force the compiler to inline the function.

The `__noinline__` and `__forceinline__` function qualifiers cannot be used together, and neither function qualifier can be applied to an inline function.