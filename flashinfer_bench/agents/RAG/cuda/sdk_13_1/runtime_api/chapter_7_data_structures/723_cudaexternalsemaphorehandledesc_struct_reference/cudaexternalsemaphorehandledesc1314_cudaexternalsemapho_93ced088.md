# cudaExternalSemaphoreHandleDesc::@13::@14 cudaExternalSemaphoreHandleDesc::win32

Win32 handle referencing the semaphore object. Valid when type is one of the following:

cudaExternalSemaphoreHandleTypeOpaqueWin32

##### **‣**

cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt

##### **‣**

cudaExternalSemaphoreHandleTypeD3D12Fence

##### **‣**

cudaExternalSemaphoreHandleTypeD3D11Fence

##### **‣**

cudaExternalSemaphoreHandleTypeKeyedMutex

##### **‣**

cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 Exactly

##### **‣**

one of 'handle' and 'name' must be non-NULL. If type is one of the
following: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
cudaExternalSemaphoreHandleTypeKeyedMutexKmt then 'name' must be NULL.