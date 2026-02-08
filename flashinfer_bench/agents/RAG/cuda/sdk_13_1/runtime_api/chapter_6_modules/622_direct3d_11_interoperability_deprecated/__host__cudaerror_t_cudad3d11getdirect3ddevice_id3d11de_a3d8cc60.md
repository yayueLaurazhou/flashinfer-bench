# __host__cudaError_t cudaD3D11GetDirect3DDevice (ID3D11Device **ppD3D11Device)

Gets the Direct3D device against which the current CUDA context was created.

##### Parameters

**ppD3D11Device**

  - Returns the Direct3D device for this thread

##### Returns

cudaSuccess, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 5.0.

This function is deprecated and should no longer be used. It is no longer necessary to associate a
CUDA device with a D3D11 device in order to achieve maximum interoperability performance.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D11SetDirect3DDevice