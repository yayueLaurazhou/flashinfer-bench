# CUresult cuExternalMemoryGetMappedBuffer (CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc)

Maps a buffer onto an imported memory object.

###### Parameters

**devPtr**

  - Returned device pointer to buffer
**extMem**

  - Handle to external memory object
**bufferDesc**

  - Buffer descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Maps a buffer onto an imported memory object and returns a device pointer in devPtr.

The properties of the buffer being mapped must be described in bufferDesc. The
CUDA_EXTERNAL_MEMORY_BUFFER_DESC structure is defined as follows:

‎    typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
unsigned long long offset;
unsigned long long size;
unsigned int flags;
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC;

where CUDA_EXTERNAL_MEMORY_BUFFER_DESC::offset is the offset in the memory object
where the buffer's base address is. CUDA_EXTERNAL_MEMORY_BUFFER_DESC::size is the size
of the buffer. CUDA_EXTERNAL_MEMORY_BUFFER_DESC::flags must be zero.

The offset and size have to be suitably aligned to match the requirements of the external API. Mapping
two buffers whose ranges overlap may or may not result in the same virtual address being returned
for the overlapped portion. In such cases, the application must ensure that all accesses to that region
from the GPU are volatile. Otherwise writes made via one address are not guaranteed to be visible via
the other address, even if they're issued by the same thread. It is recommended that applications map
the combined range instead of mapping separate buffers and then apply the appropriate offsets to the
returned pointer to derive the individual buffers.

The returned pointer devPtr must be freed using cuMemFree.


CUDA Driver API TRM-06703-001 _vRelease Version  |  363


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuImportExternalMemory, cuDestroyExternalMemory,
cuExternalMemoryGetMappedMipmappedArray