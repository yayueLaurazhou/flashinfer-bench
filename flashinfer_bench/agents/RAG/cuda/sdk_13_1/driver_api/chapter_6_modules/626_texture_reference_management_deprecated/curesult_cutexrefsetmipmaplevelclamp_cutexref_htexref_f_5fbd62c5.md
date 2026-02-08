# CUresult cuTexRefSetMipmapLevelClamp (CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)

Sets the mipmap min/max mipmap level clamps for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference


CUDA Driver API TRM-06703-001 _vRelease Version  |  523


Modules


**minMipmapLevelClamp**

  - Mipmap min level clamp
**maxMipmapLevelClamp**

  - Mipmap max level clamp

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the min/max mipmap level clamps, minMipmapLevelClamp and
maxMipmapLevelClamp respectively, to be used when reading memory through the texture
reference hTexRef.

Note that this call has no effect if hTexRef is not bound to a mipmapped array.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat