# CUresult cuLinkComplete (CUlinkState state, void **cubinOut, size_t *sizeOut)

Complete a pending linker invocation.

###### Parameters

**state**
A pending linker invocation
**cubinOut**
On success, this will point to the output image
**sizeOut**
Optional parameter to receive the size of the generated image

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Completes the pending linker action and returns the cubin image for the linked device code, which can
be used with cuModuleLoadData. The cubin is owned by state, so it should be loaded before state
is destroyed via cuLinkDestroy. This call does not destroy state.


See also:

cuLinkCreate, cuLinkAddData, cuLinkAddFile, cuLinkDestroy, cuModuleLoadData