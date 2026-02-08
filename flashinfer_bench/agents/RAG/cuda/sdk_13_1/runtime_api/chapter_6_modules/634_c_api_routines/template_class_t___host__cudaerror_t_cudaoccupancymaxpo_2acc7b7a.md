# template < class T > __host__cudaError_t cudaOccupancyMaxPotentialClusterSize (int *clusterSize, T *func, const cudaLaunchConfig_t *config)

Given the kernel function (func) and launch configuration (config), return the maximum cluster
size in *clusterSize.

##### Parameters

**clusterSize**

  - Returned maximum cluster size that can be launched for the given kernel function and launch
configuration
**func**

  - Kernel function for which maximum cluster size is calculated
**config**

  - Launch configuration for the given kernel function

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown,


CUDA Runtime API vRelease Version  |  512


Modules

##### Description

The cluster dimensions in config are ignored. If func has a required cluster size set (see
cudaFuncGetAttributes),*clusterSize will reflect the required cluster size.

By default this function will always return a value that's portable on future hardware. A higher value
may be returned if the kernel function allows non-portable cluster sizes.

This function will respect the compile time launch bounds.



See also:

cudaFuncGetAttributes