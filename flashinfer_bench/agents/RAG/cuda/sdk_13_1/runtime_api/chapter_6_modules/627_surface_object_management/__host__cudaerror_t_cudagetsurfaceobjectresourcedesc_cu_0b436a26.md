# __host__cudaError_t cudaGetSurfaceObjectResourceDesc (cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject)

Returns a surface object's resource descriptor Returns the resource descriptor for the surface object
specified by surfObject.

##### Parameters

**pResDesc**

  - Resource descriptor
**surfObject**

  - Surface object


CUDA Runtime API vRelease Version  |  314


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

See also:

cudaCreateSurfaceObject, cuSurfObjectGetResourceDesc