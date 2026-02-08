# cudaExternalMemoryHandleDesc::@11::@12 cudaExternalMemoryHandleDesc::win32

Win32 handle referencing the semaphore object. Valid when type is one of the following:

cudaExternalMemoryHandleTypeOpaqueWin32

##### **‣**

cudaExternalMemoryHandleTypeOpaqueWin32Kmt

##### **‣**

cudaExternalMemoryHandleTypeD3D12Heap

##### **‣**

cudaExternalMemoryHandleTypeD3D12Resource

##### **‣**

cudaExternalMemoryHandleTypeD3D11Resource

##### **‣**

cudaExternalMemoryHandleTypeD3D11ResourceKmt Exactly one of 'handle' and 'name' must be

##### **‣**

non-NULL. If type is one of the following: cudaExternalMemoryHandleTypeOpaqueWin32Kmt
cudaExternalMemoryHandleTypeD3D11ResourceKmt then 'name' must be NULL.