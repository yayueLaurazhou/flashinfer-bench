# __host__cudaError_t cudaArrayGetPlane (cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx)

Gets a CUDA array plane from a CUDA array.

##### Parameters

**pPlaneArray**

  - Returned CUDA array referenced by the planeIdx
**hArray**

  - CUDA array
**planeIdx**

  - Plane index

##### Returns

cudaSuccess, cudaErrorInvalidValue cudaErrorInvalidResourceHandle

##### Description

Returns in pPlaneArray a CUDA array that represents a single format plane of the CUDA array
hArray.

If planeIdx is greater than the maximum number of planes in this array or if the array does not have
a multi-planar format e.g: cudaChannelFormatKindNV12, then cudaErrorInvalidValue is returned.

Note that if the hArray has format cudaChannelFormatKindNV12, then passing in 0 for
planeIdx returns a CUDA array of the same size as hArray but with one 8-bit channel and
cudaChannelFormatKindUnsigned as its format kind. If 1 is passed for planeIdx, then the
returned CUDA array has half the height and width of hArray with two 8-bit channels and
cudaChannelFormatKindUnsigned as its format kind.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArrayGetPlane


CUDA Runtime API vRelease Version  |  120


Modules