# template < class T > __host__cudaError_t cudaLaunchKernel (T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)

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
cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion,
cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound,
cudaErrorJitCompilationDisabled

##### Description

The function invokes kernel func on gridDim (gridDim.x gridDim.y gridDim.z) grid of
blocks. Each block contains blockDim (blockDim.x blockDim.y blockDim.z) threads.

If the kernel has N parameters the args should point to array of N pointers. Each pointer, from
args[0] to args[N - 1], point to the region of memory from which the actual parameter will be
copied.

sharedMem sets the amount of dynamic shared memory that will be available to each thread block.

stream specifies a stream the invocation is associated to.


Note:


CUDA Runtime API vRelease Version  |  483


Modules













cudaLaunchKernel ( C API)