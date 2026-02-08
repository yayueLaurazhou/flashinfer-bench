# __host__cudaError_t cudaWGLGetDevice (int *device, HGPUNV hGpu)

Gets the CUDA device associated with hGpu.

##### Parameters

**device**

  - Returns the device associated with hGpu, or -1 if hGpu is not a compute device.
**hGpu**

  - Handle to a GPU, as queried via WGL_NV_gpu_affinity

##### Returns

cudaSuccess

##### Description

Returns the CUDA device associated with a hGpu, if applicable.


Note:


CUDA Runtime API vRelease Version  |  235


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

WGL_NV_gpu_affinity, cuWGLGetDevice