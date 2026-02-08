# CUresult cuDeviceGraphMemTrim (CUdevice device)

Free unused memory that was cached on the specified device for use with graphs back to the OS.

###### Parameters

**device**

  - The device for which cached memory should be freed.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE

###### Description

Blocks which are not in use by a graph that is either currently executing or scheduled to execute are
freed back to the operating system.


See also:

cuGraphAddMemAllocNode, cuGraphAddMemFreeNode, cuDeviceSetGraphMemAttribute,
cuDeviceGetGraphMemAttribute