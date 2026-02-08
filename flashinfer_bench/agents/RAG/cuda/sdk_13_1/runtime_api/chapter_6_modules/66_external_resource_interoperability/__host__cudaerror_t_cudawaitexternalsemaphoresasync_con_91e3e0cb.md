# __host__cudaError_t cudaWaitExternalSemaphoresAsync (const cudaExternalSemaphore_t *extSemArray, const cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream)

Waits on a set of external semaphore objects.

##### Parameters

**extSemArray**

  - External semaphores to be waited on
**paramsArray**

  - Array of semaphore parameters
**numExtSems**

  - Number of semaphores to wait on
**stream**

  - Stream to enqueue the wait operations in


CUDA Runtime API vRelease Version  |  93


Modules

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle cudaErrorTimeout

##### Description

Enqueues a wait operation on a set of externally allocated semaphore object in the specified stream.
The operations will be executed when all prior operations in the stream complete.

The exact semantics of waiting on a semaphore depends on the type of the object.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeOpaqueFd, cudaExternalSemaphoreHandleTypeOpaqueWin32,
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt then waiting on the semaphore will wait until
the semaphore reaches the signaled state. The semaphore will then be reset to the unsignaled state.
Therefore for every signal operation, there can only be one wait operation.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeD3D12Fence, cudaExternalSemaphoreHandleTypeD3D11Fence,
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 then waiting on
the semaphore will wait until the value of the semaphore is greater than or equal to
cudaExternalSemaphoreWaitParams::params::fence::value.

If the semaphore object is of the type cudaExternalSemaphoreHandleTypeNvSciSync then, waiting
on the semaphore will wait until the cudaExternalSemaphoreSignalParams::params::nvSciSync::fence
is signaled by the signaler of the NvSciSyncObj that was associated with this semaphore
object. By default, waiting on such an external semaphore object causes appropriate memory
synchronization operations to be performed over all external memory objects that are imported as
cudaExternalMemoryHandleTypeNvSciBuf. This ensures that any subsequent accesses made by
other importers of the same set of NvSciBuf memory object(s) are coherent. These operations can
be skipped by specifying the flag cudaExternalSemaphoreWaitSkipNvSciBufMemSync, which
can be used as a performance optimization when data coherency is not required. But specifying
this flag in scenarios where data coherency is required results in undefined behavior. Also, for
semaphore object of the type cudaExternalSemaphoreHandleTypeNvSciSync, if the NvSciSyncAttrList
used to create the NvSciSyncObj had not set the flags in cudaDeviceGetNvSciSyncAttributes to
cudaNvSciSyncAttrWait, this API will return cudaErrorNotSupported.

If the semaphore object is any one of the following types:
cudaExternalSemaphoreHandleTypeKeyedMutex,
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, then the keyed
mutex will be acquired when it is released with the key specified in
cudaExternalSemaphoreSignalParams::params::keyedmutex::key or until the timeout specified by
cudaExternalSemaphoreSignalParams::params::keyedmutex::timeoutMs has lapsed. The timeout
interval can either be a finite value specified in milliseconds or an infinite value. In case an infinite
value is specified the timeout never elapses. The windows INFINITE macro must be used to specify
infinite timeout


CUDA Runtime API vRelease Version  |  94


Modules





See also:

cudaImportExternalSemaphore, cudaDestroyExternalSemaphore,
cudaSignalExternalSemaphoresAsync