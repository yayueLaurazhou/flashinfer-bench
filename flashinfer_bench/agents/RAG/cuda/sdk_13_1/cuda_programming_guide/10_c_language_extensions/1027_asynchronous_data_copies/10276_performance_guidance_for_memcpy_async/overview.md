# 10.27.6. Performance Guidance for memcpy_async

### 10.27.6. Performance Guidance for `memcpy_async`[](#performance-guidance-for-memcpy-async "Permalink to this headline")

For compute capability 8.x, the pipeline mechanism is shared among CUDA threads in the same CUDA warp. This sharing causes batches of `memcpy_async` to be entangled within a warp, which can impact performance under certain circumstances.

This section highlights the warp-entanglement effect on *commit*, *wait*, and *arrive* operations. Please refer to [Pipeline Interface](#pipeline-interface) and the [Pipeline Primitives Interface](#pipeline-primitives-interface) for an overview of the individual operations.