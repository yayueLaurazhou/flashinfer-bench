# __host__cudaError_t cudaGraphRetainUserObject (cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags)

Retain a reference to a user object from a graph.

##### Parameters

**graph**

  - The graph to associate the reference with
**object**

  - The user object to retain a reference for
**count**

  - The number of references to add to the graph, typically 1. Must be nonzero and not larger than
INT_MAX.
**flags**

  - The optional flag cudaGraphUserObjectMove transfers references from the calling thread, rather
than create new references. Pass 0 to create new references.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates or moves user object references that will be owned by a CUDA graph.


CUDA Runtime API vRelease Version  |  424


Modules


See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cudaUserObjectCreate cudaUserObjectRetain, cudaUserObjectRelease, cudaGraphReleaseUserObject,
cudaGraphCreate