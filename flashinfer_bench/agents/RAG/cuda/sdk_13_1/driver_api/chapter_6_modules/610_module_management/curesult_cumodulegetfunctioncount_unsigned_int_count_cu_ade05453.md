# CUresult cuModuleGetFunctionCount (unsigned int *count, CUmodule mod)

Returns the number of functions within a module.

###### Parameters

**count**

  - Number of functions found within the module
**mod**

  - Module to query


CUDA Driver API TRM-06703-001 _vRelease Version  |  150


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in count the number of functions in mod.