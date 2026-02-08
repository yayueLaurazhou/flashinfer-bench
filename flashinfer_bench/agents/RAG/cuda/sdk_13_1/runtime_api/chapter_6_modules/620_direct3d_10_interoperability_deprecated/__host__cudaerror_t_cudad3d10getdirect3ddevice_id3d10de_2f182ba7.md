# __host__cudaError_t cudaD3D10GetDirect3DDevice (ID3D10Device **ppD3D10Device)

Gets the Direct3D device against which the current CUDA context was created.

##### Parameters

**ppD3D10Device**

  - Returns the Direct3D device for this thread


CUDA Runtime API vRelease Version  |  264


Modules

##### Returns

cudaSuccess, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 5.0.

This function is deprecated and should no longer be used. It is no longer necessary to associate a
CUDA device with a D3D10 device in order to achieve maximum interoperability performance.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D10SetDirect3DDevice