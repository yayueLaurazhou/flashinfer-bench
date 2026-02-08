# CUresult cuTexRefGetAddress (CUdeviceptr *pdptr, CUtexref hTexRef)

Gets the address associated with a texture reference.

###### Parameters

**pdptr**

  - Returned device address
**hTexRef**

  - Texture reference

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated


CUDA Driver API TRM-06703-001 _vRelease Version  |  506


Modules


Returns in *pdptr the base address bound to the texture reference hTexRef, or returns
CUDA_ERROR_INVALID_VALUE if the texture reference is not bound to any device memory
range.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFilterMode, cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat