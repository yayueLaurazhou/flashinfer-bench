# enum CUexternalSemaphoreHandleType

External semaphore handle types

###### Values

**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1**
Handle is an opaque file descriptor
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2**
Handle is an opaque shared NT handle
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3**
Handle is an opaque, globally shared handle
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4**
Handle is a shared NT handle referencing a D3D12 fence object
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5**
Handle is a shared NT handle referencing a D3D11 fence object
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6**
Opaque handle to NvSciSync Object
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7**
Handle is a shared NT handle referencing a D3D11 keyed mutex object
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8**
Handle is a globally shared handle referencing a D3D11 keyed mutex object
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9**


CUDA Driver API TRM-06703-001 _vRelease Version  |  42


Modules


Handle is an opaque file descriptor referencing a timeline semaphore
**CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10**
Handle is an opaque shared NT handle referencing a timeline semaphore