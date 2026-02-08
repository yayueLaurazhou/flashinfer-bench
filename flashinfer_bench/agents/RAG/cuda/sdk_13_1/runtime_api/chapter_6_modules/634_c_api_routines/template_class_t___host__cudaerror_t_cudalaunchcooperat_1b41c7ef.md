# template < class T > __host__cudaError_t cudaLaunchCooperativeKernel (T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)

Launches a device function.

##### Parameters

**func**

  - Device function symbol
**gridDim**

  - Grid dimentions
**blockDim**

  - Block dimentions
**args**

  - Arguments
**sharedMem**

  - Shared memory (defaults to 0)
**stream**

  - Stream identifier (defaults to NULL)

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources,
cudaErrorSharedObjectInitFailed

##### Description

The function invokes kernel func on gridDim (gridDim.x gridDim.y gridDim.z) grid of
blocks. Each block contains blockDim (blockDim.x blockDim.y blockDim.z) threads.


CUDA Runtime API vRelease Version  |  481


Modules


The device on which this kernel is invoked must have a non-zero value for the device attribute
cudaDevAttrCooperativeLaunch.

The total number of blocks launched cannot exceed the maximum number of blocks per
multiprocessor as returned by cudaOccupancyMaxActiveBlocksPerMultiprocessor (or
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors as
specified by the device attribute cudaDevAttrMultiProcessorCount.

The kernel cannot make use of CUDA dynamic parallelism.

If the kernel has N parameters the args should point to array of N pointers. Each pointer, from
args[0] to args[N - 1], point to the region of memory from which the actual parameter will be
copied.

sharedMem sets the amount of dynamic shared memory that will be available to each thread block.

stream specifies a stream the invocation is associated to.















cudaLaunchCooperativeKernel ( C API)


CUDA Runtime API vRelease Version  |  482


Modules