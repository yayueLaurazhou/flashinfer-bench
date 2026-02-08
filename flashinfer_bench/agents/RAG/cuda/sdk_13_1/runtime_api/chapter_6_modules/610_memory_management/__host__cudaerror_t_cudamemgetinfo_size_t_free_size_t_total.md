# __host__cudaError_t cudaMemGetInfo (size_t *free, size_t *total)

Gets free and total device memory.

##### Parameters

**free**

  - Returned free memory in bytes
**total**

  - Returned total memory in bytes

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure

##### Description

Returns in *total the total amount of memory available to the the current context. Returns in
*free the amount of memory on the device that is free according to the OS. CUDA is not guaranteed
to be able to allocate all of the memory that the OS reports as free. In a multi-tenet situation, free
estimate returned is prone to race condition where a new allocation/free done by a different process or
a different thread in the same process between the time when free memory was estimated and reported,
will result in deviation in free value reported and actual free memory.

The integrated GPU on Tegra shares memory with CPU and other component of the SoC. The free and
total values returned by the API excludes the SWAP memory space maintained by the OS on some
platforms. The OS may move some of the memory pages into swap area as the GPU or CPU allocate or
access memory. See Tegra app note on how to calculate total and free memory on Tegra.



See also:

cuMemGetInfo


CUDA Runtime API vRelease Version  |  185


Modules