# 13.6.3.3.1.3. Pending Kernel Launches (CDP1)

###### 13.6.3.3.1.3. Pending Kernel Launches (CDP1)[](#pending-kernel-launches-cdp1 "Permalink to this headline")

See [Pending Kernel Launches](#pending-kernel-launches), above, for CDP2 version of document.

When a kernel is launched, all associated configuration and parameter data is tracked until the kernel completes. This data is stored within a system-managed launch pool.

The launch pool is divided into a fixed-size pool and a virtualized pool with lower performance. The device runtime system software will try to track launch data in the fixed-size pool first. The virtualized pool will be used to track new launches when the fixed-size pool is full.

The size of the fixed-size launch pool is configurable by calling `cudaDeviceSetLimit()` from the host and specifying `cudaLimitDevRuntimePendingLaunchCount`.