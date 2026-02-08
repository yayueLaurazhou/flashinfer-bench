# template < class T > __host__cudaError_t cudaGetKernel (cudaKernel_t *kernelPtr, T *func)

Get pointer to device kernel that matches entry function entryFuncAddr.

##### Parameters

**kernelPtr**

  - Returns the device kernel
**func**

##### Returns

cudaSuccess

##### Description

Returns in kernelPtr the device kernel corresponding to the entry function entryFuncAddr.


See also:

cudaGetKernel ( C API)