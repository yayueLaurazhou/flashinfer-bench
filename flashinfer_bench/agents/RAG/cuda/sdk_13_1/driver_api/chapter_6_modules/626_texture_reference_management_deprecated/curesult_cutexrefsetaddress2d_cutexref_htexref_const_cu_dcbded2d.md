# CUresult cuTexRefSetAddress2D (CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch)

Binds an address as a 2D texture reference.

###### Parameters

**hTexRef**

  - Texture reference to bind
**desc**

  - Descriptor of CUDA array


CUDA Driver API TRM-06703-001 _vRelease Version  |  515


Modules


**dptr**

  - Device pointer to bind
**Pitch**

  - Line pitch in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Binds a linear address range to the texture reference hTexRef. Any previous address or CUDA array
state associated with the texture reference is superseded by this function. Any memory previously
bound to hTexRef is unbound.

Using a tex2D() function inside a kernel requires a call to either cuTexRefSetArray() to bind the
corresponding texture reference to an array, or cuTexRefSetAddress2D() to bind the texture reference
to linear memory.

Function calls to cuTexRefSetFormat() cannot follow calls to cuTexRefSetAddress2D() for the same
texture reference.

It is required that dptr be aligned to the appropriate hardware-specific texture alignment. You can
query this value using the device attribute CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. If
an unaligned dptr is supplied, CUDA_ERROR_INVALID_VALUE is returned.

Pitch has to be aligned to the hardware-specific texture pitch alignment. This value can be queried
using the device attribute CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. If an
unaligned Pitch is supplied, CUDA_ERROR_INVALID_VALUE is returned.

Width and Height, which are specified in elements (or texels), cannot exceed
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
respectively. Pitch, which is specified in bytes, cannot exceed
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.


See also:

cuTexRefSetAddress, cuTexRefSetAddressMode, cuTexRefSetArray, cuTexRefSetFilterMode,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat


CUDA Driver API TRM-06703-001 _vRelease Version  |  516


Modules