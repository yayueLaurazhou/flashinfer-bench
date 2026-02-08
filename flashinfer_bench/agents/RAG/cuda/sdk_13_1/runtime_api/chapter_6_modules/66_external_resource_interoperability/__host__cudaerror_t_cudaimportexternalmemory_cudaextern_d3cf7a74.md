# __host__cudaError_t cudaImportExternalMemory (cudaExternalMemory_t *extMem_out, const cudaExternalMemoryHandleDesc *memHandleDesc)

Imports an external memory object.

##### Parameters

**extMem_out**

  - Returned handle to an external memory object


CUDA Runtime API vRelease Version  |  85


Modules


**memHandleDesc**

  - Memory import handle descriptor

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorOperatingSystem

##### Description

Imports an externally allocated memory object and returns a handle to that in extMem_out.

The properties of the handle being imported must be described in memHandleDesc. The
cudaExternalMemoryHandleDesc structure is defined as follows:


where cudaExternalMemoryHandleDesc::type specifies the type of handle being imported.
cudaExternalMemoryHandleType is defined as:


If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeOpaqueFd, then
cudaExternalMemoryHandleDesc::handle::fd must be a valid file descriptor referencing a memory
object. Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported
successfully. Performing any operations on the file descriptor after it is imported results in undefined
behavior.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeOpaqueWin32,
then exactly one of cudaExternalMemoryHandleDesc::handle::win32::handle and
cudaExternalMemoryHandleDesc::handle::win32::name must not be NULL. If
cudaExternalMemoryHandleDesc::handle::win32::handle is not NULL, then it must represent a valid
shared NT handle that references a memory object. Ownership of this handle is not transferred to
CUDA after the import operation, so the application must release the handle using the appropriate
system call. If cudaExternalMemoryHandleDesc::handle::win32::name is not NULL, then it must point
to a NULL-terminated array of UTF-16 characters that refers to a memory object.


CUDA Runtime API vRelease Version  |  86


Modules


If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeOpaqueWin32Kmt,
then cudaExternalMemoryHandleDesc::handle::win32::handle must be non-NULL and
cudaExternalMemoryHandleDesc::handle::win32::name must be NULL. The handle specified must be
a globally shared KMT handle. This handle does not hold a reference to the underlying object, and thus
will be invalid when all references to the memory object are destroyed.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeD3D12Heap,
then exactly one of cudaExternalMemoryHandleDesc::handle::win32::handle and
cudaExternalMemoryHandleDesc::handle::win32::name must not be NULL. If
cudaExternalMemoryHandleDesc::handle::win32::handle is not NULL, then it must represent
a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when
referring to a ID3D12Heap object. This handle holds a reference to the underlying object. If
cudaExternalMemoryHandleDesc::handle::win32::name is not NULL, then it must point to a NULLterminated array of UTF-16 characters that refers to a ID3D12Heap object.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeD3D12Resource,
then exactly one of cudaExternalMemoryHandleDesc::handle::win32::handle and
cudaExternalMemoryHandleDesc::handle::win32::name must not be NULL. If
cudaExternalMemoryHandleDesc::handle::win32::handle is not NULL, then it must represent
a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when
referring to a ID3D12Resource object. This handle holds a reference to the underlying object. If
cudaExternalMemoryHandleDesc::handle::win32::name is not NULL, then it must point to a NULLterminated array of UTF-16 characters that refers to a ID3D12Resource object.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeD3D11Resource,then
exactly one of cudaExternalMemoryHandleDesc::handle::win32::handle and
cudaExternalMemoryHandleDesc::handle::win32::name must not be NULL. If
cudaExternalMemoryHandleDesc::handle::win32::handle is not NULL, then it must represent a valid
shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a
ID3D11Resource object. If cudaExternalMemoryHandleDesc::handle::win32::name is not NULL,
then it must point to a NULL-terminated array of UTF-16 characters that refers to a ID3D11Resource
object.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeD3D11ResourceKmt,
then cudaExternalMemoryHandleDesc::handle::win32::handle must be non-NULL and
cudaExternalMemoryHandleDesc::handle::win32::name must be NULL. The handle specified must be
a valid shared KMT handle that is returned by IDXGIResource::GetSharedHandle when referring to a
ID3D11Resource object.

If cudaExternalMemoryHandleDesc::type is cudaExternalMemoryHandleTypeNvSciBuf,
then cudaExternalMemoryHandleDesc::handle::nvSciBufObject must be NON-NULL and
reference a valid NvSciBuf object. If the NvSciBuf object imported into CUDA is also
mapped by other drivers, then the application must use cudaWaitExternalSemaphoresAsync or
cudaSignalExternalSemaphoresAsync as approprriate barriers to maintain coherence between
CUDA and the other drivers. See cudaExternalSemaphoreWaitSkipNvSciBufMemSync and
cudaExternalSemaphoreSignalSkipNvSciBufMemSync for memory synchronization.


CUDA Runtime API vRelease Version  |  87


Modules


The size of the memory object must be specified in cudaExternalMemoryHandleDesc::size.

Specifying the flag cudaExternalMemoryDedicated in cudaExternalMemoryHandleDesc::flags
indicates that the resource is a dedicated resource. The definition of what
a dedicated resource is outside the scope of this extension. This flag must
be set if cudaExternalMemoryHandleDesc::type is one of the following:
cudaExternalMemoryHandleTypeD3D12Resource cudaExternalMemoryHandleTypeD3D11Resource
cudaExternalMemoryHandleTypeD3D11ResourceKmt



See also:

cudaDestroyExternalMemory, cudaExternalMemoryGetMappedBuffer,
cudaExternalMemoryGetMappedMipmappedArray