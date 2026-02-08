# 13.6.2.1.8. API Reference (CDP1)

##### 13.6.2.1.8. API Reference (CDP1)[](#api-reference-cdp1 "Permalink to this headline")

See [API Reference](#api-reference-cdp2), above, for CDP2 version of document.

The portions of the CUDA Runtime API supported in the device runtime are detailed here. Host and device runtime APIs have identical syntax; semantics are the same except where indicated. The table below provides an overview of the API relative to the version available from the host.

Table 16 Supported API Functions[](#id474 "Permalink to this table")




| Runtime API Functions | Details |
| --- | --- |
| `cudaDeviceSynchronize` | Synchronizes on work launched from thread’s own block only.  Warning: Note that calling this API from device code is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release. |
| `cudaDeviceGetCacheConfig` |  |
| `cudaDeviceGetLimit` |  |
| `cudaGetLastError` | Last error is per-thread state, not per-block state |
| `cudaPeekAtLastError` |  |
| `cudaGetErrorString` |  |
| `cudaGetDeviceCount` |  |
| `cudaDeviceGetAttribute` | Will return attributes for any device |
| `cudaGetDevice` | Always returns current device ID as would be seen from host |
| `cudaStreamCreateWithFlags` | Must pass `cudaStreamNonBlocking` flag |
| `cudaStreamDestroy` |  |
| `cudaStreamWaitEvent` |  |
| `cudaEventCreateWithFlags` | Must pass `cudaEventDisableTiming` flag |
| `cudaEventRecord` |  |
| `cudaEventDestroy` |  |
| `cudaFuncGetAttributes` |  |
| `cudaMemcpyAsync` | Notes about all `memcpy/memset` functions:   * Only async `memcpy/set` functions are supported * Only device-to-device `memcpy` is permitted * May not pass in local or shared memory pointers |
| `cudaMemcpy2DAsync` |
| `cudaMemcpy3DAsync` |
| `cudaMemsetAsync` |
| `cudaMemset2DAsync` |  |
| `cudaMemset3DAsync` |  |
| `cudaRuntimeGetVersion` |  |
| `cudaMalloc` | May not call `cudaFree` on the device on a pointer created on the host, and vice-versa |
| `cudaFree` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |  |
| `cudaOccupancyMaxPotentialBlockSize` |  |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` |  |