# CUresult cuUserObjectCreate (CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags)

Create a user object.

###### Parameters

**object_out**

  - Location to return the user object handle
**ptr**

  - The pointer to pass to the destroy function
**destroy**

  - Callback to free the user object when it is no longer in use
**initialRefcount**

  - The initial refcount to create the object with, typically 1. The initial references are owned by the
calling thread.
**flags**

  - Currently it is required to pass CU_USER_OBJECT_NO_DESTRUCTOR_SYNC, which is the
only defined flag. This indicates that the destroy callback cannot be waited on by any CUDA API.
Users requiring synchronization of the callback should signal its completion manually.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Create a user object with the specified destructor callback and initial reference count. The initial
references are owned by the caller.

Destructor callbacks cannot make CUDA API calls and should avoid blocking behavior, as they are
executed by a shared internal thread. Another thread may be signaled to perform such actions, if it does
not block forward progress of tasks scheduled through CUDA.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cuUserObjectRetain, cuUserObjectRelease, cuGraphRetainUserObject, cuGraphReleaseUserObject,
cuGraphCreate


CUDA Driver API TRM-06703-001 _vRelease Version  |  495


Modules