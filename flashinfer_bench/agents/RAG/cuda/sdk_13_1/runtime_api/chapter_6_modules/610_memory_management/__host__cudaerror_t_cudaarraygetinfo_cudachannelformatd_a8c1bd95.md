# __host__cudaError_t cudaArrayGetInfo (cudaChannelFormatDesc *desc, cudaExtent *extent, unsigned int *flags, cudaArray_t array)

Gets info about the specified cudaArray.

##### Parameters

**desc**

  - Returned array type
**extent**

  - Returned array shape. 2D arrays will have depth of zero
**flags**

  - Returned array flags
**array**

  - The cudaArray to get info for

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns in *desc, *extent and *flags respectively, the type, shape and flags of array.

Any of *desc, *extent and *flags may be specified as NULL.


CUDA Runtime API vRelease Version  |  118


Modules





See also:

cuArrayGetDescriptor, cuArray3DGetDescriptor