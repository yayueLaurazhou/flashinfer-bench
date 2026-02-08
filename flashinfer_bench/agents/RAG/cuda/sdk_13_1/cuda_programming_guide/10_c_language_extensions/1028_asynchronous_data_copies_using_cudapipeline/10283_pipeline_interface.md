# 10.28.3. Pipeline Interface

### 10.28.3. Pipeline Interface[](#pipeline-interface "Permalink to this headline")

The complete API documentation for `cuda::memcpy_async` is provided in the [libcudacxx API](https://nvidia.github.io/libcudacxx) documentation along with some examples.

The `pipeline` interface requires

* at least CUDA 11.0,
* at least ISO C++ 2011 compatibility, e.g., to be compiled with `-std=c++11`, and
* `#include <cuda/pipeline>`.

For a C-like interface, when compiling without ISO C++ 2011 compatibility, see [Pipeline Primitives Interface](#pipeline-primitives-interface).