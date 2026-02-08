# __host__cudaError_t cudaHostUnregister (void *ptr)

Unregisters a memory range that was registered with cudaHostRegister.

##### Parameters

**ptr**

  - Host pointer to memory to unregister

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorHostMemoryNotRegistered

##### Description

Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.

The base address must be the same one specified to cudaHostRegister().





CUDA Runtime API vRelease Version  |  133


Modules


See also:

cudaHostUnregister, cuMemHostUnregister