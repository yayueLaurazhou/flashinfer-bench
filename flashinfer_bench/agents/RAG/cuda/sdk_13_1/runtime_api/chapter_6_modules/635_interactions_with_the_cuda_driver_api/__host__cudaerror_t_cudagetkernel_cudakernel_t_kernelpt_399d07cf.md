# __host__cudaError_t cudaGetKernel (cudaKernel_t *kernelPtr, const void *entryFuncAddr)

Get pointer to device kernel that matches entry function entryFuncAddr.

##### Parameters

**kernelPtr**

  - Returns the device kernel
**entryFuncAddr**

  - Address of device entry function to search kernel for

##### Returns

cudaSuccess

##### Description

Returns in kernelPtr the device kernel corresponding to the entry function entryFuncAddr.

Note that it is possible that there are multiple symbols belonging to different translation units with the
same entryFuncAddr registered with this CUDA Runtime and so the order which the translation
units are loaded and registered with the CUDA Runtime can lead to differing return pointers in
kernelPtr . Suggested methods of ensuring uniqueness are to limit visibility of __global__ device
functions by using static or hidden visibility attribute in the respective translation units.


See also:

cudaGetKernel (C++ API)