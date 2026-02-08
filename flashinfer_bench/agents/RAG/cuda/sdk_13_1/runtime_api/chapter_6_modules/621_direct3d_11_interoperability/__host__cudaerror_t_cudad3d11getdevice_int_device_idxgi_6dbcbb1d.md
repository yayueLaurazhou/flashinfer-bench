# __host__cudaError_t cudaD3D11GetDevice (int *device, IDXGIAdapter *pAdapter)

Gets the device number for an adapter.

##### Parameters

**device**

  - Returns the device corresponding to pAdapter
**pAdapter**

  - D3D11 adapter to get device for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Returns in *device the CUDA-compatible device corresponding to the adapter pAdapter obtained
from IDXGIFactory::EnumAdapters. This call will succeed only if a device on adapter pAdapter is
CUDA-compatible.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuD3D11GetDevice