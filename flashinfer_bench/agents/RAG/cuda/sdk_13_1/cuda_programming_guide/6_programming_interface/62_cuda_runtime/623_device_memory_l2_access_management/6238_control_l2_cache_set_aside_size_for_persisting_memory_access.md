# 6.2.3.8. Control L2 Cache Set-Aside Size for Persisting Memory Access

#### 6.2.3.8. Control L2 Cache Set-Aside Size for Persisting Memory Access[](#control-l2-cache-set-aside-size-for-persisting-memory-access "Permalink to this headline")

The L2 set-aside cache size for persisting memory accesses is queried using CUDA runtime API `cudaDeviceGetLimit` and set using CUDA runtime API `cudaDeviceSetLimit` as a `cudaLimit`. The maximum value for setting this limit is `cudaDeviceProp::persistingL2CacheMaxSize`.

```
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```