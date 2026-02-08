# CUresult cuCtxFromGreenCtx (CUcontext *pContext, CUgreenCtx hCtx)

Converts a green context into the primary context.

###### Parameters

**pContext**
Returned primary context with green context resources
**hCtx**
Green context to convert

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

The API converts a green context into the primary context returned in pContext. It is important
to note that the converted context pContext is a normal primary context but with the resources of
the specified green context hCtx. Once converted, it can then be used to set the context current with
cuCtxSetCurrent or with any of the CUDA APIs that accept a CUcontext parameter.

Users are expected to call this API before calling any CUDA APIs that accept a CUcontext. Failing to
do so will result in the APIs returning CUDA_ERROR_INVALID_CONTEXT.


See also:

cuGreenCtxCreate