# __host__cudaError_t cudaLaunchKernel (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)

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

  - Shared memory
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration,
cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources,


CUDA Runtime API vRelease Version  |  106


Modules


cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorUnsupportedPtxVersion,
cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound,
cudaErrorJitCompilationDisabled

##### Description

The function invokes kernel func on gridDim (gridDim.x gridDim.y gridDim.z) grid of
blocks. Each block contains blockDim (blockDim.x blockDim.y blockDim.z) threads.

If the kernel has N parameters the args should point to array of N pointers. Each pointer, from
args[0] to args[N - 1], point to the region of memory from which the actual parameter will be
copied.

For templated functions, pass the function symbol as follows:
func_name<template_arg_0,...,template_arg_N>

sharedMem sets the amount of dynamic shared memory that will be available to each thread block.

stream specifies a stream the invocation is associated to.















See also:

cudaLaunchKernel (C++ API), cuLaunchKernel


CUDA Runtime API vRelease Version  |  107


Modules