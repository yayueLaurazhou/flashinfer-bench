# 13.4.3.1.2. Pending Kernel Launches

##### 13.4.3.1.2. Pending Kernel Launches[](#pending-kernel-launches "Permalink to this headline")

When a kernel is launched, all associated configuration and parameter data is tracked until the kernel completes. This data is stored within a system-managed launch pool.

The size of the fixed-size launch pool is configurable by calling `cudaDeviceSetLimit()` from the host and specifying `cudaLimitDevRuntimePendingLaunchCount`.