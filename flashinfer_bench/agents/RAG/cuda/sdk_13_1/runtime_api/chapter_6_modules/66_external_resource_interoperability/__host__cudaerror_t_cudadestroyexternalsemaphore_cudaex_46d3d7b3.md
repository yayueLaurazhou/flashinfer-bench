# __host__cudaError_t cudaDestroyExternalSemaphore (cudaExternalSemaphore_t extSem)

Destroys an external semaphore.

##### Parameters

**extSem**

  - External semaphore to be destroyed

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle

##### Description

Destroys an external semaphore object and releases any references to the underlying resource. Any
outstanding signals or waits must have completed before the semaphore is destroyed.









See also:

cudaImportExternalSemaphore, cudaSignalExternalSemaphoresAsync,
cudaWaitExternalSemaphoresAsync


CUDA Runtime API vRelease Version  |  82


Modules