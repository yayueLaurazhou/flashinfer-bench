# __host__cudaError_t cudaCreateSurfaceObject (cudaSurfaceObject_t *pSurfObject, const cudaResourceDesc *pResDesc)

Creates a surface object.

##### Parameters

**pSurfObject**

  - Surface object to create
**pResDesc**

  - Resource descriptor

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidChannelDescriptor,
cudaErrorInvalidResourceHandle

##### Description

Creates a surface object and returns it in pSurfObject. pResDesc describes the data to
perform surface load/stores on. cudaResourceDesc::resType must be cudaResourceTypeArray and
cudaResourceDesc::res::array::array must be set to a valid CUDA array handle.

Surface objects are only supported on devices of compute capability 3.0 or higher. Additionally, a
surface object is an opaque value, and, as such, should only be accessed through CUDA API calls.





See also:

cudaDestroySurfaceObject, cuSurfObjectCreate


CUDA Runtime API vRelease Version  |  313


Modules