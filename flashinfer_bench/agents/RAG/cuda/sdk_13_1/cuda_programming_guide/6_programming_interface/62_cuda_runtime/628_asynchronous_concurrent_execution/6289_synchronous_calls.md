# 6.2.8.9. Synchronous Calls

#### 6.2.8.9. Synchronous Calls[](#synchronous-calls "Permalink to this headline")

When a synchronous function is called, control is not returned to the host thread before the device has completed the requested task. Whether the host thread will then yield, block, or spin can be specified by calling `cudaSetDeviceFlags()`with some specific flags (see reference manual for details) before any other CUDA call is performed by the host thread.