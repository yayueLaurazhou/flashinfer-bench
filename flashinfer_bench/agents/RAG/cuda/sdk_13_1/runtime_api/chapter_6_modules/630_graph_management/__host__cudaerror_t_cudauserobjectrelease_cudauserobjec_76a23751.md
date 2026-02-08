# __host__cudaError_t cudaUserObjectRelease (cudaUserObject_t object, unsigned int count)

Release a reference to a user object.

##### Parameters

**object**

  - The object to release
**count**

  - The number of references to release, typically 1. Must be nonzero and not larger than INT_MAX.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Releases user object references owned by the caller. The object's destructor is invoked if the reference
count reaches zero.

It is undefined behavior to release references not owned by the caller, or to use a user object handle
after all references are released.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cudaUserObjectCreate, cudaUserObjectRetain, cudaGraphRetainUserObject,
cudaGraphReleaseUserObject, cudaGraphCreate