# CUresult cuLinkAddFile (CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues)

Add a file input to a pending linker invocation.

###### Parameters

**state**
A pending linker action
**type**
The type of the input data
**path**
Path to the input file
**numOptions**
Size of options
**options**
Options to be applied only for this input (overrides options from cuLinkCreate)
**optionValues**
Array of option values, each cast to void *

###### Returns

CUDA_SUCCESS, CUDA_ERROR_FILE_NOT_FOUND CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_IMAGE,
CUDA_ERROR_INVALID_PTX, CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU

###### Description

No reference is retained to any inputs after this call returns.

This method accepts only compiler options, which are used if the input must be compiled from
PTX, and does not accept any of CU_JIT_WALL_TIME, CU_JIT_INFO_LOG_BUFFER,
CU_JIT_ERROR_LOG_BUFFER, CU_JIT_TARGET_FROM_CUCONTEXT, or CU_JIT_TARGET.

This method is equivalent to invoking cuLinkAddData on the contents of the file.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  146


Modules


For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted


See also:

cuLinkCreate, cuLinkAddData, cuLinkComplete, cuLinkDestroy