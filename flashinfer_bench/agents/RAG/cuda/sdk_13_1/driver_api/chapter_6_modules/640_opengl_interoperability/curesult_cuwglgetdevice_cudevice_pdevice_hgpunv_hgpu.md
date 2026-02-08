# CUresult cuWGLGetDevice (CUdevice *pDevice, HGPUNV hGpu)

Gets the CUDA device associated with hGpu.

###### Parameters

**pDevice**

  - Device associated with hGpu
**hGpu**

  - Handle to a GPU, as queried via WGL_NV_gpu_affinity()


CUDA Driver API TRM-06703-001 _vRelease Version  |  603


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *pDevice the CUDA device associated with a hGpu, if applicable.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGLMapBufferObject, cuGLRegisterBufferObject, cuGLUnmapBufferObject,
cuGLUnregisterBufferObject, cuGLUnmapBufferObjectAsync, cuGLSetBufferObjectMapFlags,
cudaWGLGetDevice