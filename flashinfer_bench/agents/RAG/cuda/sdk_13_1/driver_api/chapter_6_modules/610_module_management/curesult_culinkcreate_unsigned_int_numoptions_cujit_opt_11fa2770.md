# CUresult cuLinkCreate (unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)

Creates a pending JIT linker invocation.

###### Parameters

**numOptions**
Size of options arrays
**options**
Array of linker and compiler options


CUDA Driver API TRM-06703-001 _vRelease Version  |  147


Modules


**optionValues**
Array of option values, each cast to void *
**stateOut**
On success, this will contain a CUlinkState to specify and complete this action

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_JIT_COMPILER_NOT_FOUND

###### Description

If the call is successful, the caller owns the returned CUlinkState, which should eventually be destroyed
with cuLinkDestroy. The device code machine size (32 or 64 bit) will match the calling application.

Both linker and compiler options may be specified. Compiler options will be applied to inputs
to this linker action which must be compiled from PTX. The options CU_JIT_WALL_TIME,
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, and CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
will accumulate data until the CUlinkState is destroyed.

The data passed in via cuLinkAddData and cuLinkAddFile will be treated as relocatable (-rdc=true
to nvcc) when linking the final cubin during cuLinkComplete and will have similar consequences as
offline relocatable device code linking.

optionValues must remain valid for the life of the CUlinkState if output options are used. No other
references to inputs are maintained after this call returns.


Note:


For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuLinkAddData, cuLinkAddFile, cuLinkComplete, cuLinkDestroy