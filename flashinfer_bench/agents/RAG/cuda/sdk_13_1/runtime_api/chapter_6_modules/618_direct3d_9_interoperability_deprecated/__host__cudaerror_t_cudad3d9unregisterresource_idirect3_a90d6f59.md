# __host__cudaError_t cudaD3D9UnregisterResource (IDirect3DResource9 *pResource)

Unregisters a Direct3D resource for access by CUDA.

##### Parameters

**pResource**

  - Resource to unregister

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Unregisters the Direct3D resource pResource so it is not accessible by CUDA unless registered
again.

If pResource is not registered, then cudaErrorInvalidResourceHandle is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource