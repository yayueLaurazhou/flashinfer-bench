# CUresult cuTexRefGetBorderColor (float *pBorderColor, CUtexref hTexRef)

Gets the border color used by a texture reference.

###### Parameters

**pBorderColor**

  - Returned Type and Value of RGBA color
**hTexRef**

  - Texture reference

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  508


Modules

###### Description

Deprecated

Returns in pBorderColor, values of the RGBA color used by the texture reference hTexRef. The
color value is of type float and holds color components in the following sequence: pBorderColor[0]
holds 'R' component pBorderColor[1] holds 'G' component pBorderColor[2] holds 'B' component
pBorderColor[3] holds 'A' component


See also:

cuTexRefSetAddressMode, cuTexRefSetAddressMode, cuTexRefSetBorderColor