# __host__cudaError_t cudaDriverGetVersion (int *driverVersion)

Returns the latest version of CUDA supported by the driver.

##### Parameters

**driverVersion**

  - Returns the CUDA driver version.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns in *driverVersion the latest version of CUDA supported by the driver. The version is
returned as (1000 major + 10 minor). For example, CUDA 9.2 would be represented by 9020. If no
driver is installed, then 0 is returned as the driver version.

This function automatically returns cudaErrorInvalidValue if driverVersion is NULL.


Note:


CUDA Runtime API vRelease Version  |  315


Modules





See also:

cudaRuntimeGetVersion, cuDriverGetVersion