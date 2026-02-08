# __host__cudaError_t cudaGraphReleaseUserObject (cudaGraph_t graph, cudaUserObject_t object, unsigned int count)

Release a user object reference from a graph.

##### Parameters

**graph**

  - The graph that will release the reference
**object**

  - The user object to release a reference for
**count**

  - The number of references to release, typically 1. Must be nonzero and not larger than INT_MAX.


CUDA Runtime API vRelease Version  |  422


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Releases user object references owned by a graph.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cudaUserObjectCreate cudaUserObjectRetain, cudaUserObjectRelease, cudaGraphRetainUserObject,
cudaGraphCreate