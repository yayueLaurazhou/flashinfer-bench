# __host__cudaError_t cudaGetChannelDesc (cudaChannelFormatDesc *desc, cudaArray_const_t array)

Get the channel descriptor of an array.

##### Parameters

**desc**

  - Channel format
**array**

  - Memory array on device

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  309


Modules

##### Description

Returns in *desc the channel descriptor of the CUDA array array.





See also:

cudaCreateChannelDesc ( C API), cudaCreateTextureObject, cudaCreateSurfaceObject