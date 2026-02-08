# 10.28.4.1. memcpy_async Primitive

#### 10.28.4.1. `memcpy_async` Primitive[](#memcpy-async-primitive "Permalink to this headline")

```
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```

* Request that the following operation be submitted for asynchronous evaluation:

  ```
  size_t i = 0;
  for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
  for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
  ```
* Requirements:

  + `dst_shared` must be a pointer to the shared memory destination for the `memcpy_async`.
  + `src_global` must be a pointer to the global memory source for the `memcpy_async`.
  + `size_and_align` must be 4, 8, or 16.
  + `zfill <= size_and_align`.
  + `size_and_align` must be the alignment of `dst_shared` and `src_global`.
* It is a race condition for any thread to modify the source memory or observe the destination memory prior to waiting for the `memcpy_async` operation to complete. Between submitting a `memcpy_async` operation and waiting for its completion, any of the following actions introduces a race condition:

  + Loading from `dst_shared`.
  + Storing to `dst_shared` or `src_global`.
  + Applying an atomic update to `dst_shared` or `src_global`.