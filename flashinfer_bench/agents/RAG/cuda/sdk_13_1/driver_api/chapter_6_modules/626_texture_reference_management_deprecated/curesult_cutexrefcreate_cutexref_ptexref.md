# CUresult cuTexRefCreate (CUtexref *pTexRef)

Creates a texture reference.

###### Parameters

**pTexRef**

  - Returned texture reference

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Creates a texture reference and returns its handle in *pTexRef. Once created, the application must
call cuTexRefSetArray() or cuTexRefSetAddress() to associate the reference with allocated memory.
Other texture reference functions are used to specify the format and interpretation (addressing, filtering,
etc.) to be used when the memory is read through this texture reference.


CUDA Driver API TRM-06703-001 _vRelease Version  |  505


Modules


See also:

cuTexRefDestroy