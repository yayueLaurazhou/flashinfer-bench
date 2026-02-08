# 18.5.24.2. Structured Binding

#### 18.5.24.2. Structured Binding[](#structured-binding "Permalink to this headline")

A structured binding cannot be declared with a variable memory space specifier.

Example:

```
struct S { int x; int y; };
__device__ auto [a1, b1] = S{4,5}; // error
```