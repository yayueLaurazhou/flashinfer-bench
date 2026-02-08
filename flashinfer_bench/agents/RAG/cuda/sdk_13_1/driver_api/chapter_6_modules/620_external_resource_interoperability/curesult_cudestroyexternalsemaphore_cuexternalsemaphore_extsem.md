# CUresult cuDestroyExternalSemaphore (CUexternalSemaphore extSem)

Destroys an external semaphore.

###### Parameters

**extSem**

  - External semaphore to be destroyed

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_HANDLE

###### Description

Destroys an external semaphore object and releases any references to the underlying resource. Any
outstanding signals or waits must have completed before the semaphore is destroyed.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuImportExternalSemaphore, cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync


CUDA Driver API TRM-06703-001 _vRelease Version  |  362


Modules