# __host__cudaError_t cudaUserObjectRetain (cudaUserObject_t object, unsigned int count)

Retain a reference to a user object.

##### Parameters

**object**

  - The object to retain
**count**

  - The number of references to retain, typically 1. Must be nonzero and not larger than INT_MAX.

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  427


Modules

##### Description

Retains new references to a user object. The new references are owned by the caller.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cudaUserObjectCreate, cudaUserObjectRelease, cudaGraphRetainUserObject,
cudaGraphReleaseUserObject, cudaGraphCreate