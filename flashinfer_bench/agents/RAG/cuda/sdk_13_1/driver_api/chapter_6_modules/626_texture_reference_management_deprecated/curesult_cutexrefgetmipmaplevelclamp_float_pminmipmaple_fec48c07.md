# CUresult cuTexRefGetMipmapLevelClamp (float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef)

Gets the min/max mipmap level clamps for a texture reference.

###### Parameters

**pminMipmapLevelClamp**

  - Returned mipmap min level clamp
**pmaxMipmapLevelClamp**

  - Returned mipmap max level clamp
**hTexRef**

  - Texture reference

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Returns the min/max mipmap level clamps in pminMipmapLevelClamp and
pmaxMipmapLevelClamp that's used when reading memory through the texture reference
hTexRef.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat


CUDA Driver API TRM-06703-001 _vRelease Version  |  513


Modules