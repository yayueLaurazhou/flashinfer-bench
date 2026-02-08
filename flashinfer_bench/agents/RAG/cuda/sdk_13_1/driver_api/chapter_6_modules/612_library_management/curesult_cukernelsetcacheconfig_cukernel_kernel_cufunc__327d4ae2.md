# CUresult cuKernelSetCacheConfig (CUkernel kernel, CUfunc_cache config, CUdevice dev)

Sets the preferred cache configuration for a device kernel.

###### Parameters

**kernel**

  - Kernel to configure cache for
**config**

  - Requested cache configuration
**dev**

  - Device to set attribute of

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through
config the preferred cache configuration for the device kernel kernel on the requested device dev.
This is only a preference. The driver will use the requested configuration if possible, but it is free to
choose a different configuration if required to execute kernel. Any context-wide preference set via
cuCtxSetCacheConfig() will be overridden by this per-kernel setting.

Note that attributes set using cuFuncSetCacheConfig() will override the attribute set by this API
irrespective of whether the call to cuFuncSetCacheConfig() is made before or after this API call.


CUDA Driver API TRM-06703-001 _vRelease Version  |  165


Modules


This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a
device-side synchronization point.

The supported cache configurations are:

CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)

###### **‣**

CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache

###### **‣**

CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory

###### **‣**

CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory

###### **‣**

See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuLibraryGetKernel,
cuKernelGetFunction, cuLibraryGetModule, cuModuleGetFunction, cuFuncSetCacheConfig,
cuCtxSetCacheConfig, cuLaunchKernel