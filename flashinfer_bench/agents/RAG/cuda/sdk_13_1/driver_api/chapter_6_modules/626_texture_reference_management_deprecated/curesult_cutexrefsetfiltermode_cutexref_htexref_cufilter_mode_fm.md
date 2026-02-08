# CUresult cuTexRefSetFilterMode (CUtexref hTexRef, CUfilter_mode fm)

Sets the filtering mode for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference
**fm**

  - Filtering mode to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the filtering mode fm to be used when reading memory through the texture reference
hTexRef. CUfilter_mode_enum is defined as:


CUDA Driver API TRM-06703-001 _vRelease Version  |  519


Modules


Note that this call has no effect if hTexRef is bound to linear memory.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat