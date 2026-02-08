# 13.6.2.1.1.2. Launch Environment Configuration (CDP1)

###### 13.6.2.1.1.2. Launch Environment Configuration (CDP1)[](#launch-environment-configuration-cdp1 "Permalink to this headline")

See [Launch Environment Configuration](#launch-environment-configuration), above, for CDP2 version of document.

All global device configuration settings (for example, shared memory and L1 cache size as returned from `cudaDeviceGetCacheConfig()`, and device limits returned from `cudaDeviceGetLimit()`) will be inherited from the parent. Likewise, device limits such as stack size will remain as-configured.

For host-launched kernels, per-kernel configurations set from the host will take precedence over the global setting. These configurations will be used when the kernel is launched from the device as well. It is not possible to reconfigure a kernel’s environment from the device.