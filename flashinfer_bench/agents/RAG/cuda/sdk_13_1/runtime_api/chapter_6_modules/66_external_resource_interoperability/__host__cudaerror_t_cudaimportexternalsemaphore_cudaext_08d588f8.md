# __host__cudaError_t cudaImportExternalSemaphore (cudaExternalSemaphore_t *extSem_out, const cudaExternalSemaphoreHandleDesc *semHandleDesc)

Imports an external semaphore.

##### Parameters

**extSem_out**

  - Returned handle to an external semaphore
**semHandleDesc**

  - Semaphore import handle descriptor

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorOperatingSystem


CUDA Runtime API vRelease Version  |  88


Modules

##### Description

Imports an externally allocated synchronization object and returns a handle to that in extSem_out.

The properties of the handle being imported must be described in semHandleDesc. The
cudaExternalSemaphoreHandleDesc is defined as follows:


where cudaExternalSemaphoreHandleDesc::type specifies the type of handle being imported.
cudaExternalSemaphoreHandleType is defined as:


If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeOpaqueFd,
then cudaExternalSemaphoreHandleDesc::handle::fd must be a valid file descriptor referencing a
synchronization object. Ownership of the file descriptor is transferred to the CUDA driver when the
handle is imported successfully. Performing any operations on the file descriptor after it is imported
results in undefined behavior.

If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeOpaqueWin32,
then exactly one of cudaExternalSemaphoreHandleDesc::handle::win32::handle and
cudaExternalSemaphoreHandleDesc::handle::win32::name must not be NULL. If
cudaExternalSemaphoreHandleDesc::handle::win32::handle is not NULL, then it must represent
a valid shared NT handle that references a synchronization object. Ownership of this handle is not
transferred to CUDA after the import operation, so the application must release the handle using the
appropriate system call. If cudaExternalSemaphoreHandleDesc::handle::win32::name is not NULL,
then it must name a valid synchronization object.

If cudaExternalSemaphoreHandleDesc::type is
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, then
cudaExternalSemaphoreHandleDesc::handle::win32::handle must be non-NULL and
cudaExternalSemaphoreHandleDesc::handle::win32::name must be NULL. The handle specified must


CUDA Runtime API vRelease Version  |  89


Modules


be a globally shared KMT handle. This handle does not hold a reference to the underlying object, and
thus will be invalid when all references to the synchronization object are destroyed.

If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeD3D12Fence,
then exactly one of cudaExternalSemaphoreHandleDesc::handle::win32::handle and
cudaExternalSemaphoreHandleDesc::handle::win32::name must not be NULL. If
cudaExternalSemaphoreHandleDesc::handle::win32::handle is not NULL, then it must represent
a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when
referring to a ID3D12Fence object. This handle holds a reference to the underlying object. If
cudaExternalSemaphoreHandleDesc::handle::win32::name is not NULL, then it must name a valid
synchronization object that refers to a valid ID3D12Fence object.

If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeD3D11Fence,
then exactly one of cudaExternalSemaphoreHandleDesc::handle::win32::handle and
cudaExternalSemaphoreHandleDesc::handle::win32::name must not be NULL. If
cudaExternalSemaphoreHandleDesc::handle::win32::handle is not NULL, then it must
represent a valid shared NT handle that is returned by ID3D11Fence::CreateSharedHandle. If
cudaExternalSemaphoreHandleDesc::handle::win32::name is not NULL, then it must name a valid
synchronization object that refers to a valid ID3D11Fence object.

If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeNvSciSync, then
cudaExternalSemaphoreHandleDesc::handle::nvSciSyncObj represents a valid NvSciSyncObj.

cudaExternalSemaphoreHandleTypeKeyedMutex, then exactly one
of cudaExternalSemaphoreHandleDesc::handle::win32::handle and
cudaExternalSemaphoreHandleDesc::handle::win32::name must not be NULL. If
cudaExternalSemaphoreHandleDesc::handle::win32::handle is not NULL, then it represent a valid
shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a
IDXGIKeyedMutex object.

If cudaExternalSemaphoreHandleDesc::type is cudaExternalSemaphoreHandleTypeKeyedMutexKmt,
then cudaExternalSemaphoreHandleDesc::handle::win32::handle must be non-NULL and
cudaExternalSemaphoreHandleDesc::handle::win32::name must be NULL. The handle specified must
represent a valid KMT handle that is returned by IDXGIResource::GetSharedHandle when referring to
a IDXGIKeyedMutex object.

If cudaExternalSemaphoreHandleDesc::type is
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, then
cudaExternalSemaphoreHandleDesc::handle::fd must be a valid file descriptor referencing a
synchronization object. Ownership of the file descriptor is transferred to the CUDA driver when the
handle is imported successfully. Performing any operations on the file descriptor after it is imported
results in undefined behavior.

If cudaExternalSemaphoreHandleDesc::type is
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32, then exactly
one of cudaExternalSemaphoreHandleDesc::handle::win32::handle and
cudaExternalSemaphoreHandleDesc::handle::win32::name must not be NULL. If
cudaExternalSemaphoreHandleDesc::handle::win32::handle is not NULL, then it must represent


CUDA Runtime API vRelease Version  |  90


Modules


a valid shared NT handle that references a synchronization object. Ownership of this handle is not
transferred to CUDA after the import operation, so the application must release the handle using the
appropriate system call. If cudaExternalSemaphoreHandleDesc::handle::win32::name is not NULL,
then it must name a valid synchronization object.



See also:

cudaDestroyExternalSemaphore, cudaSignalExternalSemaphoresAsync,
cudaWaitExternalSemaphoresAsync