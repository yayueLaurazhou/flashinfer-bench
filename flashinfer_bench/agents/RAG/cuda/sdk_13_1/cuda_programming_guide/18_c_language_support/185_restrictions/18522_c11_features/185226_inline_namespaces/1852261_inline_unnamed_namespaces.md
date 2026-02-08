# 18.5.22.6.1. Inline unnamed namespaces

##### 18.5.22.6.1. Inline unnamed namespaces[](#inline-unnamed-namespaces "Permalink to this headline")

The following entities cannot be declared in namespace scope within an inline unnamed namespace:

* `__managed__`, `__device__`, `__shared__` and `__constant__` variables
* `__global__` function and function templates
* variables with surface or texture type

Example:

```
inline namespace {
  namespace N2 {
    template <typename T>
    __global__ void foo(void);            // error

    __global__ void bar(void) { }         // error

    template <>
    __global__ void foo<int>(void) { }    // error

    __device__ int x1b;                   // error
    __constant__ int x2b;                 // error
    __shared__ int x3b;                   // error

    texture<int> q2;                      // error
    surface<int> s2;                      // error
  }
};
```