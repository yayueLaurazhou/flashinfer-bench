# 10.27.1. memcpy_async API

### 10.27.1. `memcpy_async` API[](#memcpy-async-api "Permalink to this headline")

The `memcpy_async` APIs are provided in the `cuda/barrier`, `cuda/pipeline`, and `cooperative_groups/memcpy_async.h` header files.

The `cuda::memcpy_async` APIs work with `cuda::barrier` and `cuda::pipeline` synchronization primitives, while the `cooperative_groups::memcpy_async` synchronizes using `cooperative_groups::wait`.

These APIs have very similar semantics: copy objects from `src` to `dst` as-if performed by another thread which, on completion of the copy, can be synchronized through `cuda::pipeline`, `cuda::barrier`, or `cooperative_groups::wait`.

The complete API documentation of the `cuda::memcpy_async` overloads for `cuda::barrier` and `cuda::pipeline` is provided in the [libcudacxx API](https://nvidia.github.io/libcudacxx) documentation along with some examples.

The API documentation of [cooperative\_groups::memcpy\_async](#collectives-cg-memcpy-async) is provided in the [Cooperative Groups](#cooperative-groups) section.

The `memcpy_async` APIs that use [cuda::barrier](#aw-barrier) and `cuda::pipeline` require compute capability 7.0 or higher. On devices with compute capability 8.0 or higher, `memcpy_async` operations from global to shared memory can benefit from hardware acceleration.