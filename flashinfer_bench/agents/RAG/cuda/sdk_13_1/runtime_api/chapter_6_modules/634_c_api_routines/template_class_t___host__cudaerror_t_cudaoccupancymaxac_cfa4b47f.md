# template < class T > __host__cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor (int *numBlocks, T func, int blockSize, size_t dynamicSMemSize)

Returns occupancy for a device function.

##### Parameters

**numBlocks**

  - Returned occupancy


CUDA Runtime API vRelease Version  |  502


Modules


**func**

  - Kernel function for which occupancy is calulated
**blockSize**

  - Block size the kernel is intended to be launched with
**dynamicSMemSize**

  - Per-block dynamic shared memory usage intended, in bytes

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,

##### Description

Returns in *numBlocks the maximum number of active blocks per streaming multiprocessor for the
device function.







See also:

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

cudaOccupancyMaxPotentialBlockSize

cudaOccupancyMaxPotentialBlockSizeWithFlags

cudaOccupancyMaxPotentialBlockSizeVariableSMem

cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags

cudaOccupancyAvailableDynamicSMemPerBlock


CUDA Runtime API vRelease Version  |  503


Modules