# 15.9.3. cudaMemPoolReuseAllowInternalDependencies

### 15.9.3. cudaMemPoolReuseAllowInternalDependencies[](#cudamempoolreuseallowinternaldependencies "Permalink to this headline")

Failing to allocate and map more physical memory from the OS, the driver will look for memory whose availability depends on another stream’s pending progress. If such memory is found, the driver will insert the required dependency into the allocating stream and reuse the memory.

```
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// When cudaMemPoolReuseAllowInternalDependencies is enabled
// and the driver fails to allocate more physical memory, the driver may
// effectively perform a cudaStreamWaitEvent in the allocating stream
// to make sure that future work in ‘otherStream’ happens after the work
// in the original stream that would be allowed to access the original allocation.
cudaMallocAsync(&ptr2, size, otherStream);
```