# 10.1.6. __inline_hint__

### 10.1.6. \_\_inline\_hint\_\_[](#inline-hint "Permalink to this headline")

The `__inline_hint__` qualifier enables more aggressive inlining in the compiler. Unlike `__forceinline__`, it does not imply that the function is inline. It can be used to improve inlining across modules when using LTO.

Neither the `__noinline__` nor the `__forceinline__` function qualifier can be used with the `__inline_hint__` function qualifier.