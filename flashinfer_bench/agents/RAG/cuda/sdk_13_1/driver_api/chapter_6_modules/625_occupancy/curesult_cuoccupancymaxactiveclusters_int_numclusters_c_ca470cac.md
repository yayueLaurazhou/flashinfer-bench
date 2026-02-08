# CUresult cuOccupancyMaxActiveClusters (int *numClusters, CUfunction func, const CUlaunchConfig *config)

Given the kernel function (func) and launch configuration (config), return the maximum number of
clusters that could co-exist on the target device in *numClusters.

###### Parameters

**numClusters**

  - Returned maximum number of clusters that could co-exist on the target device
**func**

  - Kernel function for which maximum number of clusters are calculated
**config**

  - Launch configuration for the given kernel function

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_CLUSTER_SIZE, CUDA_ERROR_UNKNOWN

###### Description

If the function has required cluster size already set (see cudaFuncGetAttributes / cuFuncGetAttribute),
the cluster size from config must either be unspecified or match the required size. Without required
sizes, the cluster size must be specified in config, else the function will return an error.

Note that various attributes of the kernel function may affect occupancy calculation. Runtime
environment may affect how the hardware schedules the clusters, so the calculated occupancy is not
guaranteed to be achievable.

Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to


CUDA Driver API TRM-06703-001 _vRelease Version  |  500


Modules


use for calculations will either be taken from the specified stream config->hStream or the current
context in case of NULL stream.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaFuncGetAttributes, cuFuncGetAttribute