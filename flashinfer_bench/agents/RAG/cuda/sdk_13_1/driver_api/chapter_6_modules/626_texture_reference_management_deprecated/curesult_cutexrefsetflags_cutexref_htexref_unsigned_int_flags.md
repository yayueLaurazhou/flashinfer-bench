# CUresult cuTexRefSetFlags (CUtexref hTexRef, unsigned int Flags)

Sets the flags for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference
**Flags**

  - Optional flags to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies optional flags via Flags to specify the behavior of data returned through the texture
reference hTexRef. The valid flags are:

CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of having the texture

###### **‣**

promote integer data to floating point data in the range [0, 1]. Note that texture with 32-bit integer
format would not be promoted, regardless of whether or not this flag is specified;
CU_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior of having

###### **‣**

the texture coordinates range from [0, Dim) where Dim is the width or height of the CUDA array.
Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension;
CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION, which disables any trilinear filtering

###### **‣**

optimizations. Trilinear optimizations improve texture filtering performance by allowing bilinear
filtering on textures in scenarios where it can closely approximate the expected results.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFilterMode, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat


CUDA Driver API TRM-06703-001 _vRelease Version  |  520


Modules