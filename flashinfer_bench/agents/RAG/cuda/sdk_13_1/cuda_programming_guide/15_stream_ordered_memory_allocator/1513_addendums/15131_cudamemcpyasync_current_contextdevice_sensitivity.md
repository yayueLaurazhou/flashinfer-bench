# 15.13.1. cudaMemcpyAsync Current Context/Device Sensitivity

### 15.13.1. cudaMemcpyAsync Current Context/Device Sensitivity[](#cudamemcpyasync-current-context-device-sensitivity "Permalink to this headline")

In the current CUDA driver, any async `memcpy` involving memory from `cudaMallocAsync` should be done using the specified stream’s context as the calling thread’s current context. This is not necessary for `cudaMemcpyPeerAsync`, as the device primary contexts specified in the API are referenced instead of the current context.