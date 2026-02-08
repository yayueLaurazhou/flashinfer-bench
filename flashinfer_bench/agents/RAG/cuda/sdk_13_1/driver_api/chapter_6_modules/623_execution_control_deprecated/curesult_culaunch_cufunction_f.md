# CUresult cuLaunch (CUfunction f)

Launches a CUDA function.

###### Parameters

**f**

  - Kernel to launch

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
CUDA_ERROR_LAUNCH_TIMEOUT,
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

###### Description

Deprecated

Invokes the kernel f on a 1 x 1 x 1 grid of blocks. The block contains the number of threads specified
by a previous call to cuFuncSetBlockShape().

The block shape, dynamic shared memory size, and parameter information must be set using
cuFuncSetBlockShape(), cuFuncSetSharedSize(), cuParamSetSize(), cuParamSeti(), cuParamSetf(),
and cuParamSetv() prior to calling this function.

Launching a function via cuLaunchKernel() invalidates the function's block shape, dynamic shared
memory size, and parameter information. After launching via cuLaunchKernel, this state must be reinitialized prior to calling this function. Failure to do so results in undefined behavior.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetBlockShape, cuFuncSetSharedSize, cuFuncGetAttribute, cuParamSetSize, cuParamSetf,
cuParamSeti, cuParamSetv, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel


CUDA Driver API TRM-06703-001 _vRelease Version  |  407


Modules