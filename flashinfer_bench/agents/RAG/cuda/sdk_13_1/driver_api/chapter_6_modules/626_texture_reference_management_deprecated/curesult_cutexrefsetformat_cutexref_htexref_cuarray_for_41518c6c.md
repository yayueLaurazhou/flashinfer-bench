# CUresult cuTexRefSetFormat (CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)

Sets the format for a texture reference.

###### Parameters

**hTexRef**

  - Texture reference
**fmt**

  - Format to set
**NumPackedComponents**

  - Number of components per array element

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the format of the data to be read by the texture reference hTexRef. fmt and
NumPackedComponents are exactly analogous to the Format and NumChannels members of
the CUDA_ARRAY_DESCRIPTOR structure: They specify the format of each component and the
number of components per array element.


See also:

cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray,
cuTexRefSetFilterMode, cuTexRefSetFlags, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat,
cudaCreateChannelDesc