# CUresult cuWaitExternalSemaphoresAsync (const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)

Waits on a set of external semaphore objects.

###### Parameters

**extSemArray**

  - External semaphores to be waited on
**paramsArray**

  - Array of semaphore parameters
**numExtSems**

  - Number of semaphores to wait on
**stream**

  - Stream to enqueue the wait operations in

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_TIMEOUT

###### Description

Enqueues a wait operation on a set of externally allocated semaphore object in the specified stream.
The operations will be executed when all prior operations in the stream complete.

The exact semantics of waiting on a semaphore depends on the type of the object.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,


CUDA Driver API TRM-06703-001 _vRelease Version  |  374


Modules


CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT then waiting on the
semaphore will wait until the semaphore reaches the signaled state. The semaphore will then be reset to
the unsignaled state. Therefore for every signal operation, there can only be one wait operation.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 then
waiting on the semaphore will wait until the value of the semaphore is greater than or equal to
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::fence::value.

If the semaphore object is of the type
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC then, waiting on the semaphore
will wait until the
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence
is signaled by the signaler of the NvSciSyncObj that was associated with this semaphore
object. By default, waiting on such an external semaphore object causes appropriate
memory synchronization operations to be performed over all external memory objects
that are imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. This
ensures that any subsequent accesses made by other importers of the same set of NvSciBuf
memory object(s) are coherent. These operations can be skipped by specifying the flag
CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC, which can be used as
a performance optimization when data coherency is not required. But specifying this flag in scenarios
where data coherency is required results in undefined behavior. Also, for semaphore object of the
type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, if the NvSciSyncAttrList
used to create the NvSciSyncObj had not set the flags in cuDeviceGetNvSciSyncAttributes to
CUDA_NVSCISYNC_ATTR_WAIT, this API will return CUDA_ERROR_NOT_SUPPORTED.

If the semaphore object is any one of the following types:
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX,
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
then the keyed mutex will be acquired when it is released with the key specified in
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::key or until the
timeout specified by
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::timeoutMs has lapsed.
The timeout interval can either be a finite value specified in milliseconds or an infinite value. In case
an infinite value is specified the timeout never elapses. The windows INFINITE macro must be used to
specify infinite timeout.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  375


Modules


See also:

cuImportExternalSemaphore, cuDestroyExternalSemaphore, cuSignalExternalSemaphoresAsync