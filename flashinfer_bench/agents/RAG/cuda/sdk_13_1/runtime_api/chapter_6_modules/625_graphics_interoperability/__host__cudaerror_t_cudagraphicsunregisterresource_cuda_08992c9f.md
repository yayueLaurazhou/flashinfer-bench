# __host__cudaError_t cudaGraphicsUnregisterResource (cudaGraphicsResource_t resource)

Unregisters a graphics resource for access by CUDA.

##### Parameters

**resource**

  - Resource to unregister

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Unregisters the graphics resource resource so it is not accessible by CUDA unless registered again.

If resource is invalid then cudaErrorInvalidResourceHandle is returned.









See also:

cudaGraphicsD3D9RegisterResource, cudaGraphicsD3D10RegisterResource,
cudaGraphicsD3D11RegisterResource, cudaGraphicsGLRegisterBuffer,
cudaGraphicsGLRegisterImage, cuGraphicsUnregisterResource