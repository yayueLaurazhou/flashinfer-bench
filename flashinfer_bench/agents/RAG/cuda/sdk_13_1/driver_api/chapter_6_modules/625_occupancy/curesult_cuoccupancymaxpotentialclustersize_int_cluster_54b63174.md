# CUresult cuOccupancyMaxPotentialClusterSize (int *clusterSize, CUfunction func, const CUlaunchConfig *config)

Given the kernel function (func) and launch configuration (config), return the maximum cluster
size in *clusterSize.

###### Parameters

**clusterSize**

  - Returned maximum cluster size that can be launched for the given kernel function and launch
configuration
**func**

  - Kernel function for which maximum cluster size is calculated
**config**

  - Launch configuration for the given kernel function

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

The cluster dimensions in config are ignored. If func has a required cluster size set (see
cudaFuncGetAttributes / cuFuncGetAttribute),*clusterSize will reflect the required cluster size.


CUDA Driver API TRM-06703-001 _vRelease Version  |  504


Modules


By default this function will always return a value that's portable on future hardware. A higher value
may be returned if the kernel function allows non-portable cluster sizes.

This function will respect the compile time launch bounds.

Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to
use for calculations will either be taken from the specified stream config->hStream or the current
context in case of NULL stream.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaFuncGetAttributes, cuFuncGetAttribute