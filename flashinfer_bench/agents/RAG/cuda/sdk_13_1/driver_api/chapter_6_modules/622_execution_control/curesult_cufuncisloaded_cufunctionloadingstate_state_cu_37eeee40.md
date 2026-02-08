# CUresult cuFuncIsLoaded (CUfunctionLoadingState *state, CUfunction function)

Returns if the function is loaded.

###### Parameters

**state**

  - returned loading state
**function**

  - the function to check

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in state the loading state of function.


See also:

cuFuncLoad, cuModuleEnumerateFunctions