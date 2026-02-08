# CUresult cuLinkAddData (CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues)

Add an input to a pending linker invocation.

###### Parameters

**state**
A pending linker action.
**type**
The type of the input data.
**data**
The input data. PTX must be NULL-terminated.
**size**
The length of the input data.
**name**
An optional name for this input in log messages.
**numOptions**
Size of options.
**options**
Options to be applied only for this input (overrides options from cuLinkCreate).
**optionValues**
Array of option values, each cast to void *.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_PTX,
CUDA_ERROR_UNSUPPORTED_PTX_VERSION, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_NO_BINARY_FOR_GPU

###### Description

Ownership of data is retained by the caller. No reference is retained to any inputs after this call
returns.

This method accepts only compiler options, which are used if the data must be compiled from
PTX, and does not accept any of CU_JIT_WALL_TIME, CU_JIT_INFO_LOG_BUFFER,
CU_JIT_ERROR_LOG_BUFFER, CU_JIT_TARGET_FROM_CUCONTEXT, or CU_JIT_TARGET.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  145


Modules


For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted


See also:

cuLinkCreate, cuLinkAddFile, cuLinkComplete, cuLinkDestroy