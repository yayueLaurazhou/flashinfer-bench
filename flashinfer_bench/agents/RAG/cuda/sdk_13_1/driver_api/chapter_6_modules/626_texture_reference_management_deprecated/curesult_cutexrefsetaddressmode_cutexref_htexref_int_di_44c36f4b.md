# CUresult cuTexRefSetAddressMode (CUtexref hTexRef, int dim, CUaddress_mode am)

Sets the addressing mode for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference
**dim**

  - Dimension
**am**

  - Addressing mode to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the addressing mode am for the given dimension dim of the texture reference hTexRef. If
dim is zero, the addressing mode is applied to the first parameter of the functions used to fetch from


Note that this call has no effect if hTexRef is bound to linear memory. Also, if the flag,
CU_TRSF_NORMALIZED_COORDINATES, is not set, the only supported address mode is
CU_TR_ADDRESS_MODE_CLAMP.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetArray, cuTexRefSetFilterMode,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat


CUDA Driver API TRM-06703-001 _vRelease Version  |  517


Modules