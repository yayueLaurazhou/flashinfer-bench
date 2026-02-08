# __host__cudaError_t cudaD3D10UnregisterResource (ID3D10Resource *pResource)

Unregisters a Direct3D resource.

##### Parameters

**pResource**

  - Resource to unregister


CUDA Runtime API vRelease Version  |  275


Modules

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Unregisters the Direct3D resource resource so it is not accessible by CUDA unless registered again.

If pResource is not registered, then cudaErrorInvalidResourceHandle is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource