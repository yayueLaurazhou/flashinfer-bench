# CUresult cuModuleEnumerateFunctions (CUfunction *functions, unsigned int numFunctions, CUmodule mod)

Returns the function handles within a module.

###### Parameters

**functions**

  - Buffer where the function handles are returned to
**numFunctions**

  - Maximum number of function handles may be returned to the buffer
**mod**

  - Module to query from

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns in functions a maximum number of numFunctions function handles within mod. When
function loading mode is set to LAZY the function retrieved may be partially loaded. The loading
state of a function can be queried using cuFunctionIsLoaded. CUDA APIs may load the function
automatically when called with partially loaded function handle which may incur additional latency.
Alternatively, cuFunctionLoad can be used to explicitly load a function. The returned function handles
become invalid when the module is unloaded.


See also:

cuModuleGetFunction, cuModuleGetFunctionCount, cuFuncIsLoaded, cuFuncLoad


CUDA Driver API TRM-06703-001 _vRelease Version  |  149


Modules