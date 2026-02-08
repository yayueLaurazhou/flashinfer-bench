# enum cudaExternalMemoryHandleType

External memory handle types

##### Values

**cudaExternalMemoryHandleTypeOpaqueFd = 1**
Handle is an opaque file descriptor
**cudaExternalMemoryHandleTypeOpaqueWin32 = 2**
Handle is an opaque shared NT handle
**cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3**
Handle is an opaque, globally shared handle
**cudaExternalMemoryHandleTypeD3D12Heap = 4**
Handle is a D3D12 heap object
**cudaExternalMemoryHandleTypeD3D12Resource = 5**
Handle is a D3D12 committed resource
**cudaExternalMemoryHandleTypeD3D11Resource = 6**
Handle is a shared NT handle to a D3D11 resource
**cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7**
Handle is a globally shared handle to a D3D11 resource
**cudaExternalMemoryHandleTypeNvSciBuf = 8**
Handle is an NvSciBuf object