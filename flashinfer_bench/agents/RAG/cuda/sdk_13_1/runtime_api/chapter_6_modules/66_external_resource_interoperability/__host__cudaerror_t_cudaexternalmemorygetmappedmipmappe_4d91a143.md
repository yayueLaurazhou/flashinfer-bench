# __host__cudaError_t cudaExternalMemoryGetMappedMipmappedArray (cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc *mipmapDesc)

Maps a CUDA mipmapped array onto an external memory object.

##### Parameters

**mipmap**

  - Returned CUDA mipmapped array
**extMem**

  - Handle to external memory object
**mipmapDesc**

  - CUDA array descriptor

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Maps a CUDA mipmapped array onto an external object and returns a handle to it in mipmap.

The properties of the CUDA mipmapped array being mapped must be described in mipmapDesc. The
structure cudaExternalMemoryMipmappedArrayDesc is defined as follows:

‎    typedef struct cudaExternalMemoryMipmappedArrayDesc_st {
unsigned long long offset;
cudaChannelFormatDesc formatDesc;


CUDA Runtime API vRelease Version  |  84


Modules


cudaExtent extent;
~~unsigned in~~ t flags;
unsigned int numLevels;
} cudaExternalMemoryMipmappedArrayDesc;

where cudaExternalMemoryMipmappedArrayDesc::offset is the offset in the memory object where
the base level of the mipmap chain is. cudaExternalMemoryMipmappedArrayDesc::formatDesc
describes the format of the data. cudaExternalMemoryMipmappedArrayDesc::extent specifies the
dimensions of the base level of the mipmap chain. cudaExternalMemoryMipmappedArrayDesc::flags
are flags associated with CUDA mipmapped arrays. For further details, please refer
to the documentation for cudaMalloc3DArray. Note that if the mipmapped array is
bound as a color target in the graphics API, then the flag cudaArrayColorAttachment
must be specified in cudaExternalMemoryMipmappedArrayDesc::flags.
cudaExternalMemoryMipmappedArrayDesc::numLevels specifies the total number of levels in the
mipmap chain.

The returned CUDA mipmapped array must be freed using cudaFreeMipmappedArray.



See also:

cudaImportExternalMemory, cudaDestroyExternalMemory, cudaExternalMemoryGetMappedBuffer