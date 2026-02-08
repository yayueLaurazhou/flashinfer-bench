# __host____device__cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor (int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize)

Returns occupancy for a device function.

##### Parameters

**numBlocks**

  - Returned occupancy
**func**

  - Kernel function for which occupancy is calculated
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





CUDA Runtime API vRelease Version  |  113


Modules





See also:

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
cudaOccupancyMaxPotentialBlockSize ( C++ API), cudaOccupancyMaxPotentialBlockSizeWithFlags
( C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem ( C+
+ API), cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
( C++ API), cudaOccupancyAvailableDynamicSMemPerBlock (C++ API),
cuOccupancyMaxActiveBlocksPerMultiprocessor