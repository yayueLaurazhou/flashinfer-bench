# CUresult cuTexRefSetArray (CUtexref hTexRef, CUarray hArray, unsigned int Flags)

Binds an array as a texture reference.

###### Parameters

**hTexRef**

  - Texture reference to bind
**hArray**

  - Array to bind
**Flags**

 - Options (must be CU_TRSA_OVERRIDE_FORMAT)

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Binds the CUDA array hArray to the texture reference hTexRef. Any previous address or CUDA
array state associated with the texture reference is superseded by this function. Flags must be set to
CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously bound to hTexRef is unbound.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetFilterMode,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat