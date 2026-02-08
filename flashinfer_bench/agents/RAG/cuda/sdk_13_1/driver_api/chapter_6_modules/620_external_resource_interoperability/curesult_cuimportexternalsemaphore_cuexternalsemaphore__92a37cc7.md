# CUresult cuImportExternalSemaphore (CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc)

Imports an external semaphore.

###### Parameters

**extSem_out**

  - Returned handle to an external semaphore
**semHandleDesc**

  - Semaphore import handle descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OPERATING_SYSTEM

###### Description

Imports an externally allocated synchronization object and returns a handle to that in extSem_out.

The properties of the handle being imported must be described in semHandleDesc. The
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC is defined as follows:


where CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type specifies the type of handle being
imported. CUexternalSemaphoreHandleType is defined as:


CUDA Driver API TRM-06703-001 _vRelease Version  |  369


Modules


If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid file descriptor
referencing a synchronization object. Ownership of the file descriptor is transferred to the CUDA
driver when the handle is imported successfully. Performing any operations on the file descriptor after
it is imported results in undefined behavior.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
NULL. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
is not NULL, then it must represent a valid shared NT handle that references a
synchronization object. Ownership of this handle is not transferred to CUDA after the import
operation, so the application must release the handle using the appropriate system call. If
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it
must name a valid synchronization object.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle must be non-NULL
and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must be NULL.
The handle specified must be a globally shared KMT handle. This handle does not hold a reference to
the underlying object, and thus will be invalid when all references to the synchronization object are
destroyed.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE, then exactly one
of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be NULL.
If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle is not NULL, then
it must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle
when referring to a ID3D12Fence object. This handle holds a reference to the underlying object. If
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it
must name a valid synchronization object that refers to a valid ID3D12Fence object.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents
a valid shared NT handle that is returned by ID3D11Fence::CreateSharedHandle. If
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it
must name a valid synchronization object that refers to a valid ID3D11Fence object.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, then


CUDA Driver API TRM-06703-001 _vRelease Version  |  370


Modules


CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::nvSciSyncObj represents a valid
NvSciSyncObj.

CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents a valid
shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a
IDXGIKeyedMutex object. If
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it
must name a valid synchronization object that refers to a valid IDXGIKeyedMutex object.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents a valid
shared KMT handle that is returned by IDXGIResource::GetSharedHandle when referring to a
IDXGIKeyedMutex object and
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must be NULL.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD, then
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid file descriptor
referencing a synchronization object. Ownership of the file descriptor is transferred to the CUDA
driver when the handle is imported successfully. Performing any operations on the file descriptor after
it is imported results in undefined behavior.

If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32, then
exactly one of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not
be NULL. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
is not NULL, then it must represent a valid shared NT handle that references a
synchronization object. Ownership of this handle is not transferred to CUDA after the import
operation, so the application must release the handle using the appropriate system call. If
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it
must name a valid synchronization object.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDestroyExternalSemaphore, cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync


CUDA Driver API TRM-06703-001 _vRelease Version  |  371


Modules