# 10.26. Asynchronous Barrier

## 10.26. Asynchronous Barrier[](#asynchronous-barrier "Permalink to this headline")

The NVIDIA C++ standard library introduces a GPU implementation of [std::barrier](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/barrier.html). Along with the implementation of `std::barrier` the library provides extensions that allow users to specify the scope of barrier objects. The barrier API scopes are documented under [Thread Scopes](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes). Devices of compute capability 8.0 or higher provide hardware acceleration for barrier operations and integration of these barriers with the [memcpy\_async](#asynchronous-data-copies) feature. On devices with compute capability below 8.0 but starting 7.0, these barriers are available without hardware acceleration.

`nvcuda::experimental::awbarrier` is deprecated in favor of `cuda::barrier`.