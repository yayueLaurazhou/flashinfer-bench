# __host__cudaError_t cudaStreamAttachMemAsync (cudaStream_t stream, void *devPtr, size_t length, unsigned int flags)

Attach memory to a stream asynchronously.

##### Parameters

**stream**

  - Stream in which to enqueue the attach operation
**devPtr**

  - Pointer to memory (must be a pointer to managed memory or to a valid host-accessible region of
system-allocated memory)
**length**

  - Length of memory (defaults to zero)
**flags**

  - Must be one of cudaMemAttachGlobal, cudaMemAttachHost or cudaMemAttachSingle (defaults
to cudaMemAttachSingle)

##### Returns

cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Enqueues an operation in stream to specify stream association of length bytes of memory starting
from devPtr. This function is a stream-ordered operation, meaning that it is dependent on, and
will only take effect when, previous work in stream has completed. Any previous association is
automatically replaced.


CUDA Runtime API vRelease Version  |  51


Modules


devPtr must point to an one of the following types of memories:

managed memory declared using the __managed__ keyword or allocated with

##### **‣**

cudaMallocManaged.
a valid host-accessible region of system-allocated pageable memory. This type of memory may

##### **‣**

only be specified if the device associated with the stream reports a non-zero value for the device
attribute cudaDevAttrPageableMemoryAccess.

For managed allocations, length must be either zero or the entire allocation's size. Both indicate that
the entire allocation's stream association is being changed. Currently, it is not possible to change stream
association for a portion of a managed allocation.

For pageable allocations, length must be non-zero.

The stream association is specified using flags which must be one of cudaMemAttachGlobal,
cudaMemAttachHost or cudaMemAttachSingle. The default value for flags is
cudaMemAttachSingle If the cudaMemAttachGlobal flag is specified, the memory can be accessed
by any stream on any device. If the cudaMemAttachHost flag is specified, the program makes a
guarantee that it won't access the memory on the device from any stream on a device that has a zero
value for the device attribute cudaDevAttrConcurrentManagedAccess. If the cudaMemAttachSingle
flag is specified and stream is associated with a device that has a zero value for the device attribute
cudaDevAttrConcurrentManagedAccess, the program makes a guarantee that it will only access the
memory on the device from stream. It is illegal to attach singly to the NULL stream, because the
NULL stream is a virtual global stream and not a specific stream. An error will be returned in this case.

When memory is associated with a single stream, the Unified Memory system will allow CPU access
to this memory region so long as all operations in stream have completed, regardless of whether
other streams are active. In effect, this constrains exclusive ownership of the managed memory region
by an active GPU to per-stream activity instead of whole-GPU activity.

Accessing memory on the device from streams that are not associated with it will produce undefined
results. No error checking is performed by the Unified Memory system to ensure that kernels launched
into other streams do not access this region.

It is a program's responsibility to order calls to cudaStreamAttachMemAsync via events,
synchronization or other means to ensure legal access to memory at all times. Data visibility and
coherency will be changed appropriately for all kernels which follow a stream-association change.

If stream is destroyed while data is associated with it, the association is removed and the association
reverts to the default visibility of the allocation as specified at cudaMallocManaged. For __managed__
variables, the default association is always cudaMemAttachGlobal. Note that destroying a stream is an
asynchronous operation, and as a result, the change to default association won't happen until all work in
the stream has completed.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  52


Modules





See also:

cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamWaitEvent, cudaStreamSynchronize,
cudaStreamAddCallback, cudaStreamDestroy, cudaMallocManaged, cuStreamAttachMemAsync