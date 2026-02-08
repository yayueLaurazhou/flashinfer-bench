# CUresult cuTexRefSetBorderColor (CUtexref hTexRef, float *pBorderColor)

Sets the border color for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference
**pBorderColor**

  - RGBA color


CUDA Driver API TRM-06703-001 _vRelease Version  |  518


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the value of the RGBA color via the pBorderColor to the texture reference hTexRef.
The color value supports only float type and holds color components in the following sequence:
pBorderColor[0] holds 'R' component pBorderColor[1] holds 'G' component pBorderColor[2] holds 'B'
component pBorderColor[3] holds 'A' component

Note that the color values can be set only when the Address mode is set to
CU_TR_ADDRESS_MODE_BORDER using cuTexRefSetAddressMode. Applications using integer
border color values have to "reinterpret_cast" their values to float.


See also:

cuTexRefSetAddressMode, cuTexRefGetAddressMode, cuTexRefGetBorderColor