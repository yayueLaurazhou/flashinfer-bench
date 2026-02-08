# CUresult cuParamSetTexRef (CUfunction hfunc, int texunit, CUtexref hTexRef)

Adds a texture-reference to the function's argument list.

###### Parameters

**hfunc**

  - Kernel to add texture-reference to
**texunit**

  - Texture unit (must be CU_PARAM_TR_DEFAULT)
**hTexRef**

  - Texture-reference to add to argument list

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Makes the CUDA array or linear memory bound to the texture reference hTexRef available to a
device program as a texture. In this version of CUDA, the texture-reference must be obtained via
cuModuleGetTexRef() and the texunit parameter must be set to CU_PARAM_TR_DEFAULT.


CUDA Driver API TRM-06703-001 _vRelease Version  |  412


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.