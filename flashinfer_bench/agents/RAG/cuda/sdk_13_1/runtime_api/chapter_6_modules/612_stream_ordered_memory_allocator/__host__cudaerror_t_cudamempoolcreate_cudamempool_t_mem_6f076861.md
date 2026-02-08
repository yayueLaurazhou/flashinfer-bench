# __host__cudaError_t cudaMemPoolCreate (cudaMemPool_t *memPool, const cudaMemPoolProps *poolProps)

Creates a memory pool.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported

##### Description

Creates a CUDA memory pool and returns the handle in pool. The poolProps determines the
properties of the pool such as the backing device and IPC capabilities.

To create a memory pool for host memory not targeting a specific NUMA node, applications
must set set cudaMemPoolProps::cudaMemLocation::type to cudaMemLocationTypeHost.
cudaMemPoolProps::cudaMemLocation::id is ignored for such pools. Pools created with the type
cudaMemLocationTypeHost are not IPC capable and cudaMemPoolProps::handleTypes must
be 0, any other values will result in cudaErrorInvalidValue. To create a memory pool targeting a
specific host NUMA node, applications must set cudaMemPoolProps::cudaMemLocation::type to
cudaMemLocationTypeHostNuma and cudaMemPoolProps::cudaMemLocation::id must specify the
NUMA ID of the host memory node. Specifying cudaMemLocationTypeHostNumaCurrent as the
cudaMemPoolProps::cudaMemLocation::type will result in cudaErrorInvalidValue. By default, the
pool's memory will be accessible from the device it is allocated on. In the case of pools created with
cudaMemLocationTypeHostNuma or cudaMemLocationTypeHost, their default accessibility will be
from the host CPU. Applications can control the maximum size of the pool by specifying a non-zero
value for cudaMemPoolProps::maxSize. If set to 0, the maximum size of the pool will default to a
system dependent value.

Applications that intend to use CU_MEM_HANDLE_TYPE_FABRIC based memory sharing must
ensure: (1) `nvidia-caps-imex-channels` character device is created by the driver and is listed under /
proc/devices (2) have at least one IMEX channel file accessible by the user launching the application.


CUDA Runtime API vRelease Version  |  215


Modules


When exporter and importer CUDA processes have been granted access to the same IMEX channel,
they can securely share memory.

The IMEX channel security model works on a per user basis. Which means all processes under a user
can share memory if the user has access to a valid IMEX channel. When multi-user isolation is desired,
a separate IMEX channel is required for each user.

These channel files exist in /dev/nvidia-caps-imex-channels/channel* and can be created using standard
OS native calls like mknod on Linux. For example: To create channel0 with the major number from /
proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/
channel0 c <major number>=""> 0`

To create a managed memory pool, applications must set
cudaMemPoolProps:cudaMemAllocationType to cudaMemAllocationTypeManaged.
cudaMemPoolProps::cudaMemAllocationHandleType must also be set to cudaMemHandleTypeNone
since IPC is not supported. For managed memory pools, cudaMemPoolProps::cudaMemLocation will
be treated as the preferred location for all allocations created from the pool. An application can also set
cudaMemLocationTypeNone to indicate no preferred location. cudaMemPoolProps::maxSize must be
set to zero for managed memory pools. cudaMemPoolProps::usage should be zero as decompress for
managed memory is not supported. For managed memory pools, all devices on the system must have
non-zero concurrentManagedAccess. If not, this call returns cudaErrorNotSupported





See also:

cuMemPoolCreate, cudaDeviceSetMemPool, cudaMallocFromPoolAsync,
cudaMemPoolExportToShareableHandle, cudaDeviceGetDefaultMemPool, cudaDeviceGetMemPool