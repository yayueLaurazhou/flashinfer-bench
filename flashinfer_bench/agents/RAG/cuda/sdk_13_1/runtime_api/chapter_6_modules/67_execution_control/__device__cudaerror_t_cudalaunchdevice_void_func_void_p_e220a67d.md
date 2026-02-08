# __device__cudaError_t cudaLaunchDevice (void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream)

Launches a specified kernel.

##### Parameters

**func**

  - Pointer to the kernel to be launched
**parameterBuffer**

  - Holds the parameters to the launched kernel. parameterBuffer can be NULL. (Optional)
**gridDimension**

  - Specifies grid dimensions
**blockDimension**

  - Specifies block dimensions
**sharedMemSize**

  - Specifies size of shared memory
**stream**

  - Specifies the stream to be used

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorLaunchMaxDepthExceeded,
cudaErrorInvalidConfiguration, cudaErrorStartupFailure, cudaErrorLaunchPendingCountExceeded,
cudaErrorLaunchOutOfResources

##### Description

Launches a specified kernel with the specified parameter buffer. A parameter buffer can be obtained by
calling cudaGetParameterBuffer().

This is a low level API and can only be accessed from Parallel Thread Execution (PTX). CUDA user
code should use <<< >>> to launch the kernels.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


Please refer to Execution Configuration and Parameter Buffer Layout from the CUDA Programming
Guide for the detailed descriptions of launch configuration and parameter layout respectively.


See also:


CUDA Runtime API vRelease Version  |  104


Modules


cudaGetParameterBuffer