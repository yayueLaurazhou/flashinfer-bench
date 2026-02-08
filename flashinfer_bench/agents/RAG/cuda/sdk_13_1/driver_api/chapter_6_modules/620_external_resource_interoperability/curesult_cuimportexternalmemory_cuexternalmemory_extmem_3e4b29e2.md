# CUresult cuImportExternalMemory (CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc)

Imports an external memory object.

###### Parameters

**extMem_out**

  - Returned handle to an external memory object
**memHandleDesc**

  - Memory import handle descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OPERATING_SYSTEM


CUDA Driver API TRM-06703-001 _vRelease Version  |  365


Modules

###### Description

Imports an externally allocated memory object and returns a handle to that in extMem_out.

The properties of the handle being imported must be described in memHandleDesc. The
CUDA_EXTERNAL_MEMORY_HANDLE_DESC structure is defined as follows:


where CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type specifies the type of handle being
imported. CUexternalMemoryHandleType is defined as:


If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, then
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid file descriptor
referencing a memory object. Ownership of the file descriptor is transferred to the CUDA driver when
the handle is imported successfully. Performing any operations on the file descriptor after it is imported
results in undefined behavior.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it must
represent a valid shared NT handle that references a memory object. Ownership of this handle is not
transferred to CUDA after the import operation, so the application must release the handle using the
appropriate system call. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a
memory object.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT, then


CUDA Driver API TRM-06703-001 _vRelease Version  |  366


Modules


CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must be non-NULL
and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must be NULL. The
handle specified must be a globally shared KMT handle. This handle does not hold a reference to the
underlying object, and thus will be invalid when all references to the memory object are destroyed.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, then exactly one
of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it
must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle
when referring to a ID3D12Heap object. This handle holds a reference to the underlying object. If
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must
point to a NULL-terminated array of UTF-16 characters that refers to a ID3D12Heap object.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, then exactly
one of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it
must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle
when referring to a ID3D12Resource object. This handle holds a reference to the underlying object. If
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must
point to a NULL-terminated array of UTF-16 characters that refers to a ID3D12Resource object.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE, then
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must represent a valid
shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a
ID3D11Resource object. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a
ID3D11Resource object.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT,
then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
must represent a valid shared KMT handle that is returned by
IDXGIResource::GetSharedHandle when referring to a ID3D11Resource object and
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must be NULL.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF, then
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::nvSciBufObject
must be non-NULL and reference a valid NvSciBuf object. If the NvSciBuf object
imported into CUDA is also mapped by other drivers, then the application must
use cuWaitExternalSemaphoresAsync or cuSignalExternalSemaphoresAsync as


CUDA Driver API TRM-06703-001 _vRelease Version  |  367


Modules


appropriate barriers to maintain coherence between CUDA and the other drivers. See
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC and
CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC for memory
synchronization.

If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD, then
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid file descriptor
referencing a dma_buf object and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags
must be zero. Importing a dma_buf object is supported only on Tegra Jetson platform starting
with Thor series. Mapping an imported dma_buf object as CUDA mipmapped array using
cuExternalMemoryGetMappedMipmappedArray is not supported.

The size of the memory object must be specified in
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::size.

Specifying the flag CUDA_EXTERNAL_MEMORY_DEDICATED in
CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags indicates that the resource is a
dedicated resource. The definition of what a dedicated resource is outside the scope of this
extension. This flag must be set if CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
one of the following: CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.

**‣** If the Vulkan memory imported into CUDA is mapped on the CPU then the application must use
vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges as well as appropriate Vulkan
pipeline barriers to maintain coherence between CPU and GPU. For more information on these
APIs, please refer to "Synchronization and Cache Control" chapter from Vulkan specification.


See also:

cuDestroyExternalMemory, cuExternalMemoryGetMappedBuffer,
cuExternalMemoryGetMappedMipmappedArray


CUDA Driver API TRM-06703-001 _vRelease Version  |  368


Modules