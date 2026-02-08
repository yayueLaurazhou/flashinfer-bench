# CUresult cuExternalMemoryGetMappedMipmappedArray (CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc)

Maps a CUDA mipmapped array onto an external memory object.

###### Parameters

**mipmap**

  - Returned CUDA mipmapped array
**extMem**

  - Handle to external memory object
**mipmapDesc**

  - CUDA array descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Maps a CUDA mipmapped array onto an external object and returns a handle to it in mipmap.

The properties of the CUDA mipmapped array being mapped must be described in mipmapDesc. The
structure CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC is defined as follows:


where CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::offset
is the offset in the memory object where the base level of the mipmap chain is.
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc describes


CUDA Driver API TRM-06703-001 _vRelease Version  |  364


Modules


the format, dimensions and type of the base level of the mipmap chain. For further details
on these parameters, please refer to the documentation for cuMipmappedArrayCreate.
Note that if the mipmapped array is bound as a color target in the graphics API,
then the flag CUDA_ARRAY3D_COLOR_ATTACHMENT must be specified in
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc::Flags.
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels specifies the total
number of levels in the mipmap chain.

If extMem was imported from a handle of type
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, then
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels must be equal to 1.

Mapping extMem imported from a handle of type
CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD, is not supported.

The returned CUDA mipmapped array must be freed using cuMipmappedArrayDestroy.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuImportExternalMemory, cuDestroyExternalMemory, cuExternalMemoryGetMappedBuffer