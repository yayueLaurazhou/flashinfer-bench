# CUresult cuFuncLoad (CUfunction function)

Loads a function.

###### Parameters

**function**

  - the function to load

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  386


Modules

###### Description

Finalizes function loading for function. Calling this API with a fully loaded function has no effect.


See also:

cuModuleEnumerateFunctions, cuFuncIsLoaded