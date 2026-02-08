# __host__cudaError_t cudaExternalMemoryGetMappedBuffer (void **devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc *bufferDesc)

Maps a buffer onto an imported memory object.

##### Parameters

**devPtr**

  - Returned device pointer to buffer
**extMem**

  - Handle to external memory object
**bufferDesc**

  - Buffer descriptor

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Maps a buffer onto an imported memory object and returns a device pointer in devPtr.

The properties of the buffer being mapped must be described in bufferDesc. The
cudaExternalMemoryBufferDesc structure is defined as follows:

‎    typedef struct cudaExternalMemoryBufferDesc_st {
unsigned long long offset;
unsigned long long size;
unsigned int flags;
} cudaExternalMemoryBufferDesc;

where cudaExternalMemoryBufferDesc::offset is the offset in the memory object where
the buffer's base address is. cudaExternalMemoryBufferDesc::size is the size of the buffer.
cudaExternalMemoryBufferDesc::flags must be zero.

The offset and size have to be suitably aligned to match the requirements of the external API. Mapping
two buffers whose ranges overlap may or may not result in the same virtual address being returned
for the overlapped portion. In such cases, the application must ensure that all accesses to that region
from the GPU are volatile. Otherwise writes made via one address are not guaranteed to be visible via
the other address, even if they're issued by the same thread. It is recommended that applications map
the combined range instead of mapping separate buffers and then apply the appropriate offsets to the
returned pointer to derive the individual buffers.

The returned pointer devPtr must be freed using cudaFree.


CUDA Runtime API vRelease Version  |  83


Modules



See also:

cudaImportExternalMemory, cudaDestroyExternalMemory,
cudaExternalMemoryGetMappedMipmappedArray