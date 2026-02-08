# CUresult cuGraphRetainUserObject (CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags)

Retain a reference to a user object from a graph.

###### Parameters

**graph**

  - The graph to associate the reference with
**object**

  - The user object to retain a reference for
**count**

  - The number of references to add to the graph, typically 1. Must be nonzero and not larger than
INT_MAX.
**flags**

 - The optional flag CU_GRAPH_USER_OBJECT_MOVE transfers references from the calling
thread, rather than create new references. Pass 0 to create new references.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Creates or moves user object references that will be owned by a CUDA graph.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


CUDA Driver API TRM-06703-001 _vRelease Version  |  493


Modules


See also:

cuUserObjectCreate, cuUserObjectRetain, cuUserObjectRelease, cuGraphReleaseUserObject,
cuGraphCreate