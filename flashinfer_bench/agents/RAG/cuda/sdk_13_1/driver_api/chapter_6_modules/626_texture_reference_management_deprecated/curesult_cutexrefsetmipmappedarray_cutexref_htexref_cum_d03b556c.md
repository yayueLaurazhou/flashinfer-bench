# CUresult cuTexRefSetMipmappedArray (CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)

Binds a mipmapped array to a texture reference.

###### Parameters

**hTexRef**

  - Texture reference to bind
**hMipmappedArray**

  - Mipmapped array to bind
**Flags**

 - Options (must be CU_TRSA_OVERRIDE_FORMAT)

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  524


Modules

###### Description

Deprecated

Binds the CUDA mipmapped array hMipmappedArray to the texture reference hTexRef. Any
previous address or CUDA array state associated with the texture reference is superseded by this
function. Flags must be set to CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously
bound to hTexRef is unbound.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetFilterMode,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat