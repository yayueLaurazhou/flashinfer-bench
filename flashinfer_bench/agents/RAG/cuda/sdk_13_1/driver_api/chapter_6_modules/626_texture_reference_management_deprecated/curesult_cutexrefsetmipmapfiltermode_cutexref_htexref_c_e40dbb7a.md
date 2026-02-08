# CUresult cuTexRefSetMipmapFilterMode (CUtexref hTexRef, CUfilter_mode fm)

Sets the mipmap filtering mode for a texture reference.

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

Specifies the mipmap filtering mode fm to be used when reading memory through the texture reference
hTexRef. CUfilter_mode_enum is defined as:


Note that this call has no effect if hTexRef is not bound to a mipmapped array.


CUDA Driver API TRM-06703-001 _vRelease Version  |  522


Modules


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat