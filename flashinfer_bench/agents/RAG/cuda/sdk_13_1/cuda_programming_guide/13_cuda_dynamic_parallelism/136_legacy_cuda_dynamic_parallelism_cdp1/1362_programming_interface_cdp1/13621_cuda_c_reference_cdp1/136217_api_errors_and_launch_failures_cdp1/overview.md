# 13.6.2.1.7. API Errors and Launch Failures (CDP1)

##### 13.6.2.1.7. API Errors and Launch Failures (CDP1)[](#api-errors-and-launch-failures-cdp1 "Permalink to this headline")

See [API Errors and Launch Failures](#api-errors-and-launch-failures), above, for CDP2 version of document.

As usual for the CUDA runtime, any function may return an error code. The last error code returned is recorded and may be retrieved via the `cudaGetLastError()` call. Errors are recorded per-thread, so that each thread can identify the most recent error that it has generated. The error code is of type `cudaError_t`.

Similar to a host-side launch, device-side launches may fail for many reasons (invalid arguments, and so on). The user must call `cudaGetLastError()` to determine if a launch generated an error, however lack of an error after launch does not imply the child kernel completed successfully.

For device-side exceptions, for example, access to an invalid address, an error in a child grid will be returned to the host instead of being returned by the parent’s call to `cudaDeviceSynchronize()`.