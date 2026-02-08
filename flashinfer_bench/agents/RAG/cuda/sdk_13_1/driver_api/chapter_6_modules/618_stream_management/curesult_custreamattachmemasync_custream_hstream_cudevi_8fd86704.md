# CUresult cuStreamAttachMemAsync (CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)

Attach memory to a stream asynchronously.

###### Parameters

**hStream**

  - Stream in which to enqueue the attach operation


CUDA Driver API TRM-06703-001 _vRelease Version  |  332


Modules


**dptr**

  - Pointer to memory (must be a pointer to managed memory or to a valid host-accessible region of
system-allocated pageable memory)
**length**

  - Length of memory
**flags**

  - Must be one of CUmemAttach_flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Enqueues an operation in hStream to specify stream association of length bytes of memory
starting from dptr. This function is a stream-ordered operation, meaning that it is dependent on,
and will only take effect when, previous work in stream has completed. Any previous association is
automatically replaced.

dptr must point to one of the following types of memories:

managed memory declared using the __managed__ keyword or allocated with

###### **‣**

cuMemAllocManaged.
a valid host-accessible region of system-allocated pageable memory. This type of memory may

###### **‣**

only be specified if the device associated with the stream reports a non-zero value for the device
attribute CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS.

For managed allocations, length must be either zero or the entire allocation's size. Both indicate that
the entire allocation's stream association is being changed. Currently, it is not possible to change stream
association for a portion of a managed allocation.

For pageable host allocations, length must be non-zero.

The stream association is specified using flags which must be one of CUmemAttach_flags. If the
CU_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed by any stream on
any device. If the CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
that it won't access the memory on the device from any stream on a device that has a zero value
for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
If the CU_MEM_ATTACH_SINGLE flag is specified and hStream
is associated with a device that has a zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, the program makes a
guarantee that it will only access the memory on the device from hStream. It is illegal to attach singly
to the NULL stream, because the NULL stream is a virtual global stream and not a specific stream. An
error will be returned in this case.


CUDA Driver API TRM-06703-001 _vRelease Version  |  333


Modules


When memory is associated with a single stream, the Unified Memory system will allow CPU access
to this memory region so long as all operations in hStream have completed, regardless of whether
other streams are active. In effect, this constrains exclusive ownership of the managed memory region
by an active GPU to per-stream activity instead of whole-GPU activity.

Accessing memory on the device from streams that are not associated with it will produce undefined
results. No error checking is performed by the Unified Memory system to ensure that kernels launched
into other streams do not access this region.

It is a program's responsibility to order calls to cuStreamAttachMemAsync via events, synchronization
or other means to ensure legal access to memory at all times. Data visibility and coherency will be
changed appropriately for all kernels which follow a stream-association change.

If hStream is destroyed while data is associated with it, the association is removed and the
association reverts to the default visibility of the allocation as specified at cuMemAllocManaged. For
__managed__ variables, the default association is always CU_MEM_ATTACH_GLOBAL. Note that
destroying a stream is an asynchronous operation, and as a result, the change to default association
won't happen until all work in the stream has completed.





See also:

cuStreamCreate, cuStreamQuery, cuStreamSynchronize, cuStreamWaitEvent, cuStreamDestroy,
cuMemAllocManaged, cudaStreamAttachMemAsync