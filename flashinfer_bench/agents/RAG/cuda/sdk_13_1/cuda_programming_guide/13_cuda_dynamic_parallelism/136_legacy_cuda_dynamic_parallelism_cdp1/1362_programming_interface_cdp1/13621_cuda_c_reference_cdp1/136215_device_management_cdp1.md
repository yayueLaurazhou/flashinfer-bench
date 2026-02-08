# 13.6.2.1.5. Device Management (CDP1)

##### 13.6.2.1.5. Device Management (CDP1)[](#device-management-programming-cdp1 "Permalink to this headline")

See [Device Management](#device-management-programming), above, for CDP2 version of document.

Only the device on which a kernel is running will be controllable from that kernel. This means that device APIs such as `cudaSetDevice()` are not supported by the device runtime. The active device as seen from the GPU (returned from `cudaGetDevice()`) will have the same device number as seen from the host system. The `cudaDeviceGetAttribute()` call may request information about another device as this API allows specification of a device ID as a parameter of the call. Note that the catch-all `cudaGetDeviceProperties()` API is not offered by the device runtime - properties must be queried individually.