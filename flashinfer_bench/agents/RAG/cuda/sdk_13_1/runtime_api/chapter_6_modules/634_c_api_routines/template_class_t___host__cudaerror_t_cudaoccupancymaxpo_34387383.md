# template < class T > __host__cudaError_t cudaOccupancyMaxPotentialBlockSize (int *minGridSize, int *blockSize, T func, size_t dynamicSMemSize, int blockSizeLimit)

Returns grid and block size that achieves maximum potential occupancy for a device function.

##### Parameters

**minGridSize**

  - Returned minimum grid size needed to achieve the best potential occupancy
**blockSize**

  - Returned block size


CUDA Runtime API vRelease Version  |  506


Modules


**func**

  - Device function symbol
**dynamicSMemSize**

  - Per-block dynamic shared memory usage intended, in bytes
**blockSizeLimit**

  - The maximum block size func is designed to work with. 0 means no limit.

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,

##### Description

Returns in *minGridSize and *blocksize a suggested grid / block size pair that achieves the
best potential occupancy (i.e. the maximum number of active warps with the smallest number of
blocks).

Use

See also:

cudaOccupancyMaxPotentialBlockSizeVariableSMem if the amount of per-block dynamic shared
memory changes with different block sizes.



See also:

cudaOccupancyMaxPotentialBlockSizeWithFlags

cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

cudaOccupancyMaxPotentialBlockSizeVariableSMem

cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags

cudaOccupancyAvailableDynamicSMemPerBlock


CUDA Runtime API vRelease Version  |  507


Modules