# enum CUpointer_attribute

Pointer information

###### Values

**CU_POINTER_ATTRIBUTE_CONTEXT = 1**
The CUcontext on which a pointer was allocated or registered
**CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2**
The CUmemorytype describing the physical location of a pointer
**CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3**
The address at which a pointer's memory may be accessed on the device
**CU_POINTER_ATTRIBUTE_HOST_POINTER = 4**
The address at which a pointer's memory may be accessed on the host
**CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5**
A pair of tokens for use with the nv-p2p.h Linux kernel interface
**CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6**
Synchronize every synchronous memory operation initiated on this region
**CU_POINTER_ATTRIBUTE_BUFFER_ID = 7**
A process-wide unique ID for an allocated memory region
**CU_POINTER_ATTRIBUTE_IS_MANAGED = 8**
Indicates if the pointer points to managed memory
**CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9**
A device ordinal of a device on which a pointer was allocated or registered
**CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10**
1 if this pointer maps to an allocation that is suitable for cudaIpcGetMemHandle, 0 otherwise
**CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11**
Starting address for this requested pointer
**CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12**
Size of the address range for this requested pointer
**CU_POINTER_ATTRIBUTE_MAPPED = 13**
1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise
**CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14**
Bitmask of allowed CUmemAllocationHandleType for this allocation
**CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15**
1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API
**CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16**


CUDA Driver API TRM-06703-001 _vRelease Version  |  69


Modules


Returns the access flags the device associated with the current context has on the corresponding
memory referenced by the pointer given
**CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17**
Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise
returns NULL.
**CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18**
Size of the actual underlying mapping that the pointer belongs to
**CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19**
The start address of the mapping that the pointer belongs to
**CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20**
A process-wide unique id corresponding to the physical allocation the pointer belongs to
**CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21**
Returns in *data a boolean that indicates whether the pointer points to memory that is capable to
be used for hardware accelerated decompression.