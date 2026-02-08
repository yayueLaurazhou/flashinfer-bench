# template < typename UnaryFunction, class T > __host__cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (int *minGridSize, int *blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit, unsigned int flags)

Returns grid and block size that achieves maximum potential occupancy for a device function.

##### Parameters

**minGridSize**

  - Returned minimum grid size needed to achieve the best potential occupancy
**blockSize**

  - Returned block size
**func**

  - Device function symbol
**blockSizeToDynamicSMemSize**

  - A unary function / functor that takes block size, and returns the size, in bytes, of dynamic shared
memory needed for a block
**blockSizeLimit**

  - The maximum block size func is designed to work with. 0 means no limit.
**flags**

  - Requested behavior for the occupancy calculator

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,


CUDA Runtime API vRelease Version  |  509


Modules

##### Description

Returns in *minGridSize and *blocksize a suggested grid / block size pair that achieves the
best potential occupancy (i.e. the maximum number of active warps with the smallest number of
blocks).



The flags parameter controls how special cases are handled. Valid flags include:

cudaOccupancyDefault: keeps the default behavior as

##### **‣**

cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags

cudaOccupancyDisableCachingOverride: This flag suppresses the default behavior on platform

##### **‣**

where global caching affects occupancy. On such platforms, if caching is enabled, but per-block
SM resource usage would result in zero occupancy, the occupancy calculator will calculate the
occupancy as if caching is disabled. Setting this flag makes the occupancy calculator to return 0 in
such cases. More information can be found about this feature in the "Unified L1/Texture Cache"
section of the Maxwell tuning guide.



See also:

cudaOccupancyMaxPotentialBlockSizeVariableSMem

cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

cudaOccupancyMaxPotentialBlockSize

cudaOccupancyMaxPotentialBlockSizeWithFlags

cudaOccupancyAvailableDynamicSMemPerBlock


CUDA Runtime API vRelease Version  |  510


Modules