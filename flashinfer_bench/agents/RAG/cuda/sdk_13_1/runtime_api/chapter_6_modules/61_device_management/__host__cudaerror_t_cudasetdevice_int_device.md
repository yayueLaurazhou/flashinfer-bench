# __host__cudaError_t cudaSetDevice (int device)

Set device to be used for GPU executions.

##### Parameters

**device**

  - Device on which the active host thread should execute the device code.

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceUnavailable,

##### Description

Sets device as the current device for the calling host thread. Valid device id's are 0 to
(cudaGetDeviceCount() - 1).

Any device memory subsequently allocated from this host thread using cudaMalloc(),
cudaMallocPitch() or cudaMallocArray() will be physically resident on device. Any host memory
allocated from this host thread using cudaMallocHost() or cudaHostAlloc() or cudaHostRegister() will
have its lifetime associated with device. Any streams or events created from this host thread will be
associated with device. Any kernels launched from this host thread using the <<<>>> operator or
cudaLaunchKernel() will be executed on device.

This call may be made from any host thread, to any device, and at any time. This function will do
no synchronization with the previous or new device, and should only take significant time when it
initializes the runtime's context state. This call will bind the primary context of the specified device to
the calling thread and all the subsequent memory allocations, stream and event creations, and kernel
launches will be associated with the primary context. This function will also immediately initialize the
runtime state on the primary context, and the context will be current on device immediately. This


CUDA Runtime API vRelease Version  |  40


Modules


function will return an error if the device is in cudaComputeModeExclusiveProcess and is occupied by
another process or if the device is in cudaComputeModeProhibited.

It is not required to call cudaInitDevice before using this function.



See also:

cudaGetDeviceCount, cudaGetDevice, cudaGetDeviceProperties, cudaChooseDevice, cudaInitDevice,
cuCtxSetCurrent