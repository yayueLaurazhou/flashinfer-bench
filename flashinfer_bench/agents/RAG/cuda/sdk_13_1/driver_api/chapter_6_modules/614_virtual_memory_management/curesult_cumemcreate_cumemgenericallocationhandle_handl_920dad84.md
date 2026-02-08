# CUresult cuMemCreate (CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags)

Create a CUDA memory handle representing a memory allocation of a given size described by the
given properties.

###### Parameters

**handle**

  - Value of handle returned. All operations on this allocation are to be performed using this handle.
**size**

  - Size of the allocation requested
**prop**

  - Properties of the allocation to create.
**flags**

  - flags for future use, must be zero now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

This creates a memory allocation on the target device specified through the prop
structure. The created allocation will not have any device or host mappings. The generic
memory handle for the allocation can be mapped to the address space of calling
process via cuMemMap. This handle cannot be transmitted directly to other processes
(see cuMemExportToShareableHandle). On Windows, the caller must also pass an
LPSECURITYATTRIBUTE in prop to be associated with this handle which limits or allows
access to this handle for a recipient process (see CUmemAllocationProp::win32HandleMetaData
for more). The size of this allocation must be a multiple of the the value given via
cuMemGetAllocationGranularity with the CU_MEM_ALLOC_GRANULARITY_MINIMUM
flag. To create a CPU allocation that doesn't target any specific NUMA nodes, applications must
set CUmemAllocationProp::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST.
CUmemAllocationProp::CUmemLocation::id is ignored for HOST allocations. HOST
allocations are not IPC capable and CUmemAllocationProp::requestedHandleTypes
must be 0, any other value will result in CUDA_ERROR_INVALID_VALUE. To
create a CPU allocation targeting a specific host NUMA node, applications must set
CUmemAllocationProp::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST_NUMA
and CUmemAllocationProp::CUmemLocation::id must specify the NUMA ID of the CPU. On


CUDA Driver API TRM-06703-001 _vRelease Version  |  273


Modules


systems where NUMA is not available CUmemAllocationProp::CUmemLocation::id must be set to 0.
Specifying CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT as the CUmemLocation::type
will result in CUDA_ERROR_INVALID_VALUE.

Applications that intend to use CU_MEM_HANDLE_TYPE_FABRIC based memory sharing must
ensure: (1) `nvidia-caps-imex-channels` character device is created by the driver and is listed under /
proc/devices (2) have at least one IMEX channel file accessible by the user launching the application.

When exporter and importer CUDA processes have been granted access to the same IMEX channel,
they can securely share memory.

The IMEX channel security model works on a per user basis. Which means all processes under a user
can share memory if the user has access to a valid IMEX channel. When multi-user isolation is desired,
a separate IMEX channel is required for each user.

These channel files exist in /dev/nvidia-caps-imex-channels/channel* and can be created using standard
OS native calls like mknod on Linux. For example: To create channel0 with the major number from /
proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/
channel0 c <major number>=""> 0`

If CUmemAllocationProp::allocFlags::usage contains CU_MEM_CREATE_USAGE_TILE_POOL
flag then the memory allocation is intended only to be used as backing tile pool for sparse CUDA
arrays and sparse CUDA mipmapped arrays. (see cuMemMapArrayAsync).


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemRelease, cuMemExportToShareableHandle, cuMemImportFromShareableHandle