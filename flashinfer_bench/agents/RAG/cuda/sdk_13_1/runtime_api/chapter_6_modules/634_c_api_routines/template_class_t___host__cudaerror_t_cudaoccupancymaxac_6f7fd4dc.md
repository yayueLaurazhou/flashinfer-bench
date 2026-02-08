# template < class T > __host__cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (int *numBlocks, T func, int blockSize, size_t dynamicSMemSize, unsigned int flags)

Returns occupancy for a device function with the specified flags.

##### Parameters

**numBlocks**

  - Returned occupancy
**func**

  - Kernel function for which occupancy is calulated
**blockSize**

  - Block size the kernel is intended to be launched with
**dynamicSMemSize**

  - Per-block dynamic shared memory usage intended, in bytes
**flags**

  - Requested behavior for the occupancy calculator

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,

##### Description

Returns in *numBlocks the maximum number of active blocks per streaming multiprocessor for the
device function.

The flags parameter controls how special cases are handled. Valid flags include:

cudaOccupancyDefault: keeps the default behavior as

##### **‣**

cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaOccupancyDisableCachingOverride: suppresses the default behavior on platform where global

##### **‣**

caching affects occupancy. On such platforms, if caching is enabled, but per-block SM resource
usage would result in zero occupancy, the occupancy calculator will calculate the occupancy as
if caching is disabled. Setting this flag makes the occupancy calculator to return 0 in such cases.
More information can be found about this feature in the "Unified L1/Texture Cache" section of the
Maxwell tuning guide.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  504


Modules





See also:

cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaOccupancyMaxPotentialBlockSize

cudaOccupancyMaxPotentialBlockSizeWithFlags

cudaOccupancyMaxPotentialBlockSizeVariableSMem

cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags

cudaOccupancyAvailableDynamicSMemPerBlock