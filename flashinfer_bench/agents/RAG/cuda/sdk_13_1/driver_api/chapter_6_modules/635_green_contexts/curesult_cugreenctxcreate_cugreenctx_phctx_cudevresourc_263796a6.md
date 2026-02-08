# CUresult cuGreenCtxCreate (CUgreenCtx *phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags)

Creates a green context with a specified set of resources.

###### Parameters

**phCtx**

  - Pointer for the output handle to the green context
**desc**

  - Descriptor generated via cuDevResourceGenerateDesc which contains the set of resources to be
used
**dev**

  - Device on which to create the green context.
**flags**

  - One of the supported green context creation flags. CU_GREEN_CTX_DEFAULT_STREAM is
required.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

This API creates a green context with the resources specified in the descriptor desc and returns it in
the handle represented by phCtx. This API will retain the primary context on device dev, which will
is released when the green context is destroyed. It is advised to have the primary context active before
calling this API to avoid the heavy cost of triggering primary context initialization and deinitialization
multiple times.

The API does not set the green context current. In order to set it current, you need to explicitly
set it current by first converting the green context to a CUcontext using cuCtxFromGreenCtx and
subsequently calling cuCtxSetCurrent / cuCtxPushCurrent. It should be noted that a green context
can be current to only one thread at a time. There is no internal synchronization to make API calls
accessing the same green context from multiple threads work.

Note: The API is not supported on 32-bit platforms.

The supported flags are:

: Creates a default stream to use inside the green context.

###### ‣ CU_GREEN_CTX_DEFAULT_STREAM

Required.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  583


Modules


cuGreenCtxDestroy, cuCtxFromGreenCtx, cuCtxSetCurrent, cuCtxPushCurrent,
cuDevResourceGenerateDesc, cuDevicePrimaryCtxRetain, cuCtxCreate