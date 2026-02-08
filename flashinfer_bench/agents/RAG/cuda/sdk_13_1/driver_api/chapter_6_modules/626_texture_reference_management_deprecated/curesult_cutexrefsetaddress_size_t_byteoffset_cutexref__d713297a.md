# CUresult cuTexRefSetAddress (size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)

Binds an address as a texture reference.

###### Parameters

**ByteOffset**

  - Returned byte offset
**hTexRef**

  - Texture reference to bind
**dptr**

  - Device pointer to bind


CUDA Driver API TRM-06703-001 _vRelease Version  |  514


Modules


**bytes**

  - Size of memory to bind in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Binds a linear address range to the texture reference hTexRef. Any previous address or CUDA array
state associated with the texture reference is superseded by this function. Any memory previously
bound to hTexRef is unbound.

Since the hardware enforces an alignment requirement on texture base addresses,
cuTexRefSetAddress() passes back a byte offset in *ByteOffset that must be applied to texture
fetches in order to read from the desired memory. This offset must be divided by the texel size and
passed to kernels that read from the texture so they can be applied to the tex1Dfetch() function.

If the device memory pointer was returned from cuMemAlloc(), the offset is guaranteed to be 0 and
NULL may be passed as the ByteOffset parameter.

The total number of elements (or texels) in the linear address range cannot exceed
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of
elements is computed as (bytes / bytesPerElement), where bytesPerElement is determined from the
data format and number of components set using cuTexRefSetFormat().


See also:

cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray, cuTexRefSetFilterMode,
cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefGetAddress, cuTexRefGetAddressMode,
cuTexRefGetArray, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat