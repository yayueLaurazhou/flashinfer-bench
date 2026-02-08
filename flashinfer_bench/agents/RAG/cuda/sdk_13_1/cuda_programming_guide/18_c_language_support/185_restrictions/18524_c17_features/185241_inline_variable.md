# 18.5.24.1. Inline Variable

#### 18.5.24.1. Inline Variable[](#inline-variable "Permalink to this headline")

* A namespace scope inline variable declared with `__device__` or `__constant__` or `__managed__` memory space specifier must have internal linkage, if the code is compiled with nvcc in whole program compilation mode.

  Examples:

  ```
  inline __device__ int xxx; //error when compiled with nvcc in
                             //whole program compilation mode.
                             //ok when compiled with nvcc in
                             //separate compilation mode.

  inline __shared__ int yyy0; // ok.

  static inline __device__ int yyy; // ok: internal linkage
  namespace {
  inline __device__ int zzz; // ok: internal linkage
  }
  ```
* When using g++ host compiler, an inline variable declared with `__managed__` memory space specifier may not be visible to the debugger.