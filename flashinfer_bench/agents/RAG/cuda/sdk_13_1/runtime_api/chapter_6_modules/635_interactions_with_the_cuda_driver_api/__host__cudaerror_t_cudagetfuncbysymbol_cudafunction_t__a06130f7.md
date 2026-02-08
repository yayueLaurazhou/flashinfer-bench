# __host__cudaError_t cudaGetFuncBySymbol (cudaFunction_t *functionPtr, const void *symbolPtr)

Get pointer to device entry function that matches entry function symbolPtr.

##### Parameters

**functionPtr**

  - Returns the device entry function
**symbolPtr**

  - Pointer to device entry function to search for

##### Returns

cudaSuccess

##### Description

Returns in functionPtr the device entry function corresponding to the symbol symbolPtr.


CUDA Runtime API vRelease Version  |  518


Modules