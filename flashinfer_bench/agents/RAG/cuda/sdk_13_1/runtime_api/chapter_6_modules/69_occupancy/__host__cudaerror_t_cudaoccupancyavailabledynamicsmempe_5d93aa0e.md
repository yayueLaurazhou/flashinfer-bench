# __host__cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock (size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize)

Returns dynamic shared memory available per block when launching numBlocks blocks on SM.

##### Parameters

**dynamicSmemSize**

  - Returned maximum dynamic shared memory
**func**

  - Kernel function for which occupancy is calculated
**numBlocks**

  - Number of blocks to fit on SM
**blockSize**

  - Size of the block

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,
cudaErrorUnknown,

##### Description

Returns in *dynamicSmemSize the maximum size of dynamic shared memory to allow
numBlocks blocks per SM.







CUDA Runtime API vRelease Version  |  112


Modules


See also:

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
cudaOccupancyMaxPotentialBlockSize ( C++ API), cudaOccupancyMaxPotentialBlockSizeWithFlags
( C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem ( C++ API),
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags ( C++ API),
cudaOccupancyAvailableDynamicSMemPerBlock