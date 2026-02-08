# CUresult cuMemAllocManaged (CUdeviceptr *dptr, size_t bytesize, unsigned int flags)

Allocates memory that will be automatically managed by the Unified Memory system.

###### Parameters

**dptr**

  - Returned device pointer
**bytesize**

  - Requested allocation size in bytes
**flags**

 - Must be one of CU_MEM_ATTACH_GLOBAL or CU_MEM_ATTACH_HOST

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Allocates bytesize bytes of managed memory on the device and returns in *dptr a
pointer to the allocated memory. If the device doesn't support allocating managed memory,
CUDA_ERROR_NOT_SUPPORTED is returned. Support for managed memory can be queried
using the device attribute CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY. The allocated
memory is suitably aligned for any kind of variable. The memory is not cleared. If bytesize is 0,
cuMemAllocManaged returns CUDA_ERROR_INVALID_VALUE. The pointer is valid on the CPU
and on all GPUs in the system that support managed memory. All accesses to this pointer must obey
the Unified Memory programming model.

flags specifies the default stream association for this allocation. flags must
be one of CU_MEM_ATTACH_GLOBAL or CU_MEM_ATTACH_HOST. If
CU_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from any
stream on any device. If CU_MEM_ATTACH_HOST is specified, then the allocation
should not be accessed from devices that have a zero value for the device attribute


CUDA Driver API TRM-06703-001 _vRelease Version  |  197


Modules


CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS; an explicit call to
cuStreamAttachMemAsync will be required to enable access on such devices.

If the association is later changed via cuStreamAttachMemAsync to a single stream, the default
association as specified during cuMemAllocManaged is restored when that stream is destroyed. For
__managed__ variables, the default association is always CU_MEM_ATTACH_GLOBAL. Note that
destroying a stream is an asynchronous operation, and as a result, the change to default association
won't happen until all work in the stream has completed.

Memory allocated with cuMemAllocManaged should be released with cuMemFree.

Device memory oversubscription is possible for GPUs that have a non-zero value for the device
attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Managed memory on
such GPUs may be evicted from device memory to host memory at any time by the Unified Memory
driver in order to make room for other allocations.

In a system where all GPUs have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, managed memory may
not be populated when this API returns and instead may be populated on access. In such systems,
managed memory can migrate to any processor's memory at any time. The Unified Memory driver will
employ heuristics to maintain data locality and prevent excessive page faults to the extent possible. The
application can also guide the driver about memory usage patterns via cuMemAdvise. The application
can also explicitly migrate memory to a desired processor's memory via cuMemPrefetchAsync.

In a multi-GPU system where all of the GPUs have a zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS and all the GPUs have peer-topeer support with each other, the physical storage for managed memory is created on the GPU which
is active at the time cuMemAllocManaged is called. All other GPUs will reference the data at reduced
bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate memory
among such GPUs.

In a multi-GPU system where not all GPUs have peer-to-peer support with each other and where the
value of the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
is zero for at least one of those GPUs, the location chosen for physical storage of managed memory is
system-dependent.

On Linux, the location chosen will be device memory as long as the current set of active contexts

###### **‣**

are on devices that either have peer-to-peer support with each other or have a non-zero value for
the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If
there is an active context on a GPU that does not have a non-zero value for that device attribute
and it does not have peer-to-peer support with the other devices that have active contexts on them,
then the location for physical storage will be 'zero-copy' or host memory. Note that this means
that managed memory that is located in device memory is migrated to host memory if a new
context is created on a GPU that doesn't have a non-zero value for the device attribute and does not
support peer-to-peer with at least one of the other devices that has an active context. This in turn
implies that context creation may fail if there is insufficient host memory to migrate all managed
allocations.


CUDA Driver API TRM-06703-001 _vRelease Version  |  198


Modules


On Windows, the physical storage is always created in 'zero-copy' or host memory. All GPUs

###### **‣**

will reference the data at reduced bandwidth over the PCIe bus. In these circumstances, use
of the environment variable CUDA_VISIBLE_DEVICES is recommended to restrict CUDA
to only use those GPUs that have peer-to-peer support. Alternatively, users can also set
CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero value to force the driver to always
use device memory for physical storage. When this environment variable is set to a non-zero value,
all contexts created in that process on devices that support managed memory have to be peerto-peer compatible with each other. Context creation will fail if a context is created on a device
that supports managed memory and is not peer-to-peer compatible with any of the other managed
memory supporting devices on which contexts were previously created, even if those contexts have
been destroyed. These environment variables are described in the CUDA programming guide under
the "CUDA environment variables" section.
On ARM, managed memory is not available on discrete gpu with Drive PX-2.

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemcpyDtoH,
cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cuDeviceGetAttribute,
cuStreamAttachMemAsync, cudaMallocManaged