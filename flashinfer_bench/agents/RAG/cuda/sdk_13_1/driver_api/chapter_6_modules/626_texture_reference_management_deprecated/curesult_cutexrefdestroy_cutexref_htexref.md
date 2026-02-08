# CUresult cuTexRefDestroy (CUtexref hTexRef)

Destroys a texture reference.

###### Parameters

**hTexRef**

  - Texture reference to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Destroys the texture reference specified by hTexRef.


See also:

cuTexRefCreate