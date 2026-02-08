# __host__cudaError_t cudaSignalExternalSemaphoresAsync (const cudaExternalSemaphore_t *extSemArray, const cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream)

Signals a set of external semaphore objects.

##### Parameters

**extSemArray**

  - Set of external semaphores to be signaled
**paramsArray**

  - Array of semaphore parameters
**numExtSems**

  - Number of semaphores to signal
**stream**

  - Stream to enqueue the signal operations in

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle

##### Description

Enqueues a signal operation on a set of externally allocated semaphore object in the specified stream.
The operations will be executed when all prior operations in the stream complete.


CUDA Runtime API vRelease Version  |  91


Modules


The exact semantics of signaling a semaphore depends on the type of the object.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeOpaqueFd, cudaExternalSemaphoreHandleTypeOpaqueWin32,
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt then signaling the semaphore will set it to the
signaled state.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeD3D12Fence, cudaExternalSemaphoreHandleTypeD3D11Fence,
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 then the semaphore will be set to the
value specified in cudaExternalSemaphoreSignalParams::params::fence::value.

If the semaphore object is of the type cudaExternalSemaphoreHandleTypeNvSciSync this
API sets cudaExternalSemaphoreSignalParams::params::nvSciSync::fence to a value that
can be used by subsequent waiters of the same NvSciSync object to order operations with
those currently submitted in stream. Such an update will overwrite previous contents of
cudaExternalSemaphoreSignalParams::params::nvSciSync::fence. By default, signaling such an
external semaphore object causes appropriate memory synchronization operations to be performed
over all the external memory objects that are imported as cudaExternalMemoryHandleTypeNvSciBuf.
This ensures that any subsequent accesses made by other importers of the same set of NvSciBuf
memory object(s) are coherent. These operations can be skipped by specifying the flag
cudaExternalSemaphoreSignalSkipNvSciBufMemSync, which can be used as a performance
optimization when data coherency is not required. But specifying this flag in scenarios
where data coherency is required results in undefined behavior. Also, for semaphore object
of the type cudaExternalSemaphoreHandleTypeNvSciSync, if the NvSciSyncAttrList used
to create the NvSciSyncObj had not set the flags in cudaDeviceGetNvSciSyncAttributes to
cudaNvSciSyncAttrSignal, this API will return cudaErrorNotSupported.

cudaExternalSemaphoreSignalParams::params::nvSciSync::fence associated with semaphore
object of the type cudaExternalSemaphoreHandleTypeNvSciSync can be deterministic.
For this the NvSciSyncAttrList used to create the semaphore object must have value of
NvSciSyncAttrKey_RequireDeterministicFences key set to true. Deterministic fences allow users to
enqueue a wait over the semaphore object even before corresponding signal is enqueued. For such
a semaphore object, CUDA guarantees that each signal operation will increment the fence value by
'1'. Users are expected to track count of signals enqueued on the semaphore object and insert waits
accordingly. When such a semaphore object is signaled from multiple streams, due to concurrent
stream execution, it is possible that the order in which the semaphore gets signaled is indeterministic.
This could lead to waiters of the semaphore getting unblocked incorrectly. Users are expected to
handle such situations, either by not using the same semaphore object with deterministic fence support
enabled in different streams or by adding explicit dependency amongst such streams so that the
semaphore is signaled in order. cudaExternalSemaphoreSignalParams::params::nvSciSync::fence
associated with semaphore object of the type cudaExternalSemaphoreHandleTypeNvSciSync
can be timestamp enabled. For this the NvSciSyncAttrList used to create the object must have
the value of NvSciSyncAttrKey_WaiterRequireTimestamps key set to true. Timestamps are
emitted asynchronously by the GPU and CUDA saves the GPU timestamp in the corresponding


CUDA Runtime API vRelease Version  |  92


Modules


NvSciSyncFence at the time of signal on GPU. Users are expected to convert GPU clocks to CPU
clocks using appropriate scaling functions. Users are expected to wait for the completion of the fence
before extracting timestamp using appropriate NvSciSync APIs. Users are expected to ensure that there
is only one outstanding timestamp enabled fence per Cuda-NvSciSync object at any point of time,
failing which leads to undefined behavior. Extracting the timestamp before the corresponding fence
is signalled could lead to undefined behaviour. Timestamp extracted via appropriate NvSciSync API
would be in microseconds.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeKeyedMutex,
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, then the keyed mutex will be released with the
key specified in cudaExternalSemaphoreSignalParams::params::keyedmutex::key.



See also:

cudaImportExternalSemaphore, cudaDestroyExternalSemaphore, cudaWaitExternalSemaphoresAsync