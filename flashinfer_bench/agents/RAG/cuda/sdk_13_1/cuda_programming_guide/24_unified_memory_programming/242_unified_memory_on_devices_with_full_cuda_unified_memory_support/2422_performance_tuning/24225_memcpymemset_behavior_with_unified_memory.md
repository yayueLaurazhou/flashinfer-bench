# 24.2.2.5. Memcpy()/Memset() Behavior With Unified Memory

#### 24.2.2.5. Memcpy()/Memset() Behavior With Unified Memory[](#memcpy-memset-behavior-with-unified-memory "Permalink to this headline")

`cudaMemcpy*()` and `cudaMemset*()` accept any unified memory pointer as arguments.

For `cudaMemcpy*()`, the direction specified as `cudaMemcpyKind` is a performance hint,
which can have a higher performance impact if any of the arguments is a unified memory pointer.

Thus, it is recommended to follow the following performance advice:

* When the physical location of unified memory is known, use an accurate `cudaMemcpyKind` hint.
* Prefer `cudaMemcpyDefault` over an inaccurate `cudaMemcpyKind` hint.
* Always use populated (initialized) buffers: avoid using these APIs to initialize memory.
* Avoid using `cudaMemcpy*()` if both pointers point to System-Allocated Memory:
  launch a kernel or use a CPU memory copy algorithm such as `std::memcpy` instead.