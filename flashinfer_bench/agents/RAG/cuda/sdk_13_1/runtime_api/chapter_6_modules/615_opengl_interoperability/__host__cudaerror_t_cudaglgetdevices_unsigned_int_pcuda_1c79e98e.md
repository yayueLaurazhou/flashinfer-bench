# __host__cudaError_t cudaGLGetDevices (unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList)

Gets the CUDA devices associated with the current OpenGL context.

##### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to the current OpenGL context
**pCudaDevices**

  - Returned CUDA devices corresponding to the current OpenGL context
**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**deviceList**

  - The set of devices to return. This set may be cudaGLDeviceListAll for all devices,
cudaGLDeviceListCurrentFrame for the devices used to render the current frame (in SLI), or
cudaGLDeviceListNextFrame for the devices used to render the next frame (in SLI).

##### Returns

cudaSuccess, cudaErrorNoDevice, cudaErrorInvalidGraphicsContext, cudaErrorOperatingSystem,
cudaErrorUnknown

##### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible devices corresponding to
the current OpenGL context. Also returns in *pCudaDevices at most cudaDeviceCount
of the CUDA-compatible devices corresponding to the current OpenGL context. If any of the
GPUs being used by the current OpenGL context are not CUDA capable then the call will return
cudaErrorNoDevice.


Note:

**‣** This function is not supported on Mac OS X.

**‣** Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuGLGetDevices


CUDA Runtime API vRelease Version  |  232


Modules