# enum CUexternalMemoryHandleType

External memory handle types


CUDA Driver API TRM-06703-001 _vRelease Version  |  41


Modules

###### Values

**CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1**
Handle is an opaque file descriptor
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2**
Handle is an opaque shared NT handle
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3**
Handle is an opaque, globally shared handle
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4**
Handle is a D3D12 heap object
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5**
Handle is a D3D12 committed resource
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6**
Handle is a shared NT handle to a D3D11 resource
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7**
Handle is a globally shared handle to a D3D11 resource
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8**
Handle is an NvSciBuf object
**CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD = 9**
Handle is a dma_buf file descriptor