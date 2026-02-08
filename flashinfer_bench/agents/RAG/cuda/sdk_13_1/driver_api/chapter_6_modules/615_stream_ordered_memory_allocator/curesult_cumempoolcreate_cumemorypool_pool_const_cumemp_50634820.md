# CUresult cuMemPoolCreate (CUmemoryPool *pool, const CUmemPoolProps *poolProps)

Creates a memory pool.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Creates a CUDA memory pool and returns the handle in pool. The poolProps determines the
properties of the pool such as the backing device and IPC capabilities.

To create a memory pool for HOST memory not targeting a specific NUMA node, applications
must set set CUmemPoolProps::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST.
CUmemPoolProps::CUmemLocation::id is ignored for such pools. Pools created with the type
CU_MEM_LOCATION_TYPE_HOST are not IPC capable and CUmemPoolProps::handleTypes
must be 0, any other values will result in CUDA_ERROR_INVALID_VALUE. To
create a memory pool targeting a specific host NUMA node, applications must set
CUmemPoolProps::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST_NUMA
and CUmemPoolProps::CUmemLocation::id must specify the NUMA ID of the host memory
node. Specifying CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT as the
CUmemPoolProps::CUmemLocation::type will result in CUDA_ERROR_INVALID_VALUE.
By default, the pool's memory will be accessible from the device it is allocated on.
In the case of pools created with CU_MEM_LOCATION_TYPE_HOST_NUMA or
CU_MEM_LOCATION_TYPE_HOST, their default accessibility will be from the host CPU.
Applications can control the maximum size of the pool by specifying a non-zero value for
CUmemPoolProps::maxSize. If set to 0, the maximum size of the pool will default to a system
dependent value.

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


CUDA Driver API TRM-06703-001 _vRelease Version  |  291


Modules


proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/
channel0 c <major number>=""> 0`

To create a managed memory pool, applications must set CUmemPoolProps::CUmemAllocationType
to CU_MEM_ALLOCATION_TYPE_MANAGED.
CUmemPoolProps::CUmemAllocationHandleType must also be set to
CU_MEM_HANDLE_TYPE_NONE since IPC is not supported. For managed memory pools,
CUmemPoolProps::CUmemLocation will be treated as the preferred location for all allocations created
from the pool. An application can also set CU_MEM_LOCATION_TYPE_NONE to indicate no
preferred location. CUmemPoolProps::maxSize must be set to zero for managed memory pools.
CUmemPoolProps::usage should be zero as decompress for managed memory is not supported. For
managed memory pools, all devices on the system must have non-zero concurrentManagedAccess. If
not, this call returns CUDA_ERROR_NOT_SUPPORTED


Note:


Specifying CU_MEM_HANDLE_TYPE_NONE creates a memory pool that will not support IPC.


See also:

cuDeviceSetMemPool, cuDeviceGetMemPool, cuDeviceGetDefaultMemPool,
cuMemAllocFromPoolAsync, cuMemPoolExportToShareableHandle