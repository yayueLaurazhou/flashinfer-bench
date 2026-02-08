# 10.35.3. Associated Host-Side API

### 10.35.3. Associated Host-Side API[](#associated-host-side-api "Permalink to this headline")

The following API functions get and set the size of the buffer used to transfer the `printf()` arguments and internal metadata to the host (default is 1 megabyte):

* `cudaDeviceGetLimit(size_t* size,cudaLimitPrintfFifoSize)`
* `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)`