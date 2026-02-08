# __host__cudaError_t cudaOccupancyMaxPotentialClusterSize (int *clusterSize, const void *func, const cudaLaunchConfig_t *launchConfig)

Given the kernel function (func) and launch configuration (config), return the maximum cluster
size in *clusterSize.

##### Parameters

**clusterSize**

  - Returned maximum cluster size that can be launched for the given kernel function and launch
configuration
**func**

  - Kernel function for which maximum cluster size is calculated
**launchConfig**

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown,

##### Description

The cluster dimensions in config are ignored. If func has a required cluster size set (see
cudaFuncGetAttributes),*clusterSize will reflect the required cluster size.

By default this function will always return a value that's portable on future hardware. A higher value
may be returned if the kernel function allows non-portable cluster sizes.

This function will respect the compile time launch bounds.





CUDA Runtime API vRelease Version  |  117


Modules





See also:

cudaFuncGetAttributes cudaOccupancyMaxPotentialClusterSize (C++ API),
cuOccupancyMaxPotentialClusterSize