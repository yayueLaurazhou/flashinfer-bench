# template < typename UnaryFunction, class T > __host__cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem (int *minGridSize, int *blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit)

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

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,

##### Description

Returns in *minGridSize and *blocksize a suggested grid / block size pair that achieves the
best potential occupancy (i.e. the maximum number of active warps with the smallest number of
blocks).



See also:


CUDA Runtime API vRelease Version  |  508


Modules


cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags

cudaOccupancyMaxActiveBlocksPerMultiprocessor

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

cudaOccupancyMaxPotentialBlockSize

cudaOccupancyMaxPotentialBlockSizeWithFlags

cudaOccupancyAvailableDynamicSMemPerBlock