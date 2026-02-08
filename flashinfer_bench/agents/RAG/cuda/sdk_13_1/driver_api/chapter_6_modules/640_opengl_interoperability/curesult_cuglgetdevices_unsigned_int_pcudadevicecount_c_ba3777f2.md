# CUresult cuGLGetDevices (unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList)

Gets the CUDA devices associated with the current OpenGL context.

###### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices.
**pCudaDevices**

  - Returned CUDA devices.
**cudaDeviceCount**

  - The size of the output device array pCudaDevices.
**deviceList**

  - The set of devices to return.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NO_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
CUDA_ERROR_OPERATING_SYSTEM

###### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible devices corresponding
to the current OpenGL context. Also returns in *pCudaDevices at most cudaDeviceCount
of the CUDA-compatible devices corresponding to the current OpenGL context. If any of the
GPUs being used by the current OpenGL context are not CUDA capable then the call will return
CUDA_ERROR_NO_DEVICE.

The deviceList argument may be any of the following:

CU_GL_DEVICE_LIST_ALL: Query all devices used by the current OpenGL context.

###### **‣**

CU_GL_DEVICE_LIST_CURRENT_FRAME: Query the devices used by the current OpenGL

###### **‣**

context to render the current frame (in SLI).
CU_GL_DEVICE_LIST_NEXT_FRAME: Query the devices used by the current OpenGL context

###### **‣**

to render the next frame (in SLI). Note that this is a prediction, it can't be guaranteed that this is
correct in all cases.


CUDA Driver API TRM-06703-001 _vRelease Version  |  600


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuWGLGetDevice, cudaGLGetDevices