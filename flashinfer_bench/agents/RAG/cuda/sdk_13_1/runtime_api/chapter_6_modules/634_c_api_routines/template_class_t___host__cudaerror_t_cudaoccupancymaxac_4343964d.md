# template < class T > __host__cudaError_t cudaOccupancyMaxActiveClusters (int *numClusters, T *func, const cudaLaunchConfig_t *config)

Given the kernel function (func) and launch configuration (config), return the maximum number of
clusters that could co-exist on the target device in *numClusters.

##### Parameters

**numClusters**

  - Returned maximum number of clusters that could co-exist on the target device
**func**

  - Kernel function for which maximum number of clusters are calculated
**config**

  - Launch configuration for the given kernel function

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorInvalidClusterSize,
cudaErrorUnknown,


CUDA Runtime API vRelease Version  |  505


Modules

##### Description

If the function has required cluster size already set (see cudaFuncGetAttributes), the cluster size from
config must either be unspecified or match the required size. Without required sizes, the cluster size
must be specified in config, else the function will return an error.

Note that various attributes of the kernel function may affect occupancy calculation. Runtime
environment may affect how the hardware schedules the clusters, so the calculated occupancy is not
guaranteed to be achievable.



See also:

cudaFuncGetAttributes