# enum cudaExternalSemaphoreHandleType

External semaphore handle types


CUDA Runtime API vRelease Version  |  554


Modules

##### Values

**cudaExternalSemaphoreHandleTypeOpaqueFd = 1**
Handle is an opaque file descriptor
**cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2**
Handle is an opaque shared NT handle
**cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3**
Handle is an opaque, globally shared handle
**cudaExternalSemaphoreHandleTypeD3D12Fence = 4**
Handle is a shared NT handle referencing a D3D12 fence object
**cudaExternalSemaphoreHandleTypeD3D11Fence = 5**
Handle is a shared NT handle referencing a D3D11 fence object
**cudaExternalSemaphoreHandleTypeNvSciSync = 6**
Opaque handle to NvSciSync Object
**cudaExternalSemaphoreHandleTypeKeyedMutex = 7**
Handle is a shared NT handle referencing a D3D11 keyed mutex object
**cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8**
Handle is a shared KMT handle referencing a D3D11 keyed mutex object
**cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9**
Handle is an opaque handle file descriptor referencing a timeline semaphore
**cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10**
Handle is an opaque handle file descriptor referencing a timeline semaphore