# CUresult cuSignalExternalSemaphoresAsync (const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)

Signals a set of external semaphore objects.

###### Parameters

**extSemArray**

  - Set of external semaphores to be signaled
**paramsArray**

  - Array of semaphore parameters
**numExtSems**

  - Number of semaphores to signal
**stream**

  - Stream to enqueue the signal operations in

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Enqueues a signal operation on a set of externally allocated semaphore object in the specified stream.
The operations will be executed when all prior operations in the stream complete.

The exact semantics of signaling a semaphore depends on the type of the object.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT then signaling the
semaphore will set it to the signaled state.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 then the
semaphore will be set to the value specified in
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::fence::value.


CUDA Driver API TRM-06703-001 _vRelease Version  |  372


Modules


If the semaphore object is of the type
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC this API sets
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence to a
value that can be used by subsequent waiters of the same NvSciSync object to order operations
with those currently submitted in stream. Such an update will overwrite previous contents
of CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence.
By default, signaling such an external semaphore object causes appropriate memory
synchronization operations to be performed over all external memory objects that are
imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. This ensures
that any subsequent accesses made by other importers of the same set of NvSciBuf
memory object(s) are coherent. These operations can be skipped by specifying the flag
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC, which can
be used as a performance optimization when data coherency is not required. But specifying
this flag in scenarios where data coherency is required results in undefined behavior. Also, for
semaphore object of the type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC,
if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
cuDeviceGetNvSciSyncAttributes to CUDA_NVSCISYNC_ATTR_SIGNAL, this API will
return CUDA_ERROR_NOT_SUPPORTED. NvSciSyncFence associated with semaphore
object of the type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC can be
deterministic. For this the NvSciSyncAttrList used to create the semaphore object must have value
of NvSciSyncAttrKey_RequireDeterministicFences key set to true. Deterministic fences allow
users to enqueue a wait over the semaphore object even before corresponding signal is enqueued.
For such a semaphore object, CUDA guarantees that each signal operation will increment the fence
value by '1'. Users are expected to track count of signals enqueued on the semaphore object and
insert waits accordingly. When such a semaphore object is signaled from multiple streams, due
to concurrent stream execution, it is possible that the order in which the semaphore gets signaled
is indeterministic. This could lead to waiters of the semaphore getting unblocked incorrectly.
Users are expected to handle such situations, either by not using the same semaphore object
with deterministic fence support enabled in different streams or by adding explicit dependency
amongst such streams so that the semaphore is signaled in order. NvSciSyncFence associated with
semaphore object of the type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC
can be timestamp enabled. For this the NvSciSyncAttrList used to create the object must have
the value of NvSciSyncAttrKey_WaiterRequireTimestamps key set to true. Timestamps are
emitted asynchronously by the GPU and CUDA saves the GPU timestamp in the corresponding
NvSciSyncFence at the time of signal on GPU. Users are expected to convert GPU clocks to CPU
clocks using appropriate scaling functions. Users are expected to wait for the completion of the fence
before extracting timestamp using appropriate NvSciSync APIs. Users are expected to ensure that there
is only one outstanding timestamp enabled fence per Cuda-NvSciSync object at any point of time,
failing which leads to undefined behavior. Extracting the timestamp before the corresponding fence
is signalled could lead to undefined behaviour. Timestamp extracted via appropriate NvSciSync API
would be in microseconds.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX,


CUDA Driver API TRM-06703-001 _vRelease Version  |  373


Modules


CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
then the keyed mutex will be released with the key specified in
CUDA_EXTERNAL_SEMAPHORE_PARAMS::params::keyedmutex::key.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuImportExternalSemaphore, cuDestroyExternalSemaphore, cuWaitExternalSemaphoresAsync