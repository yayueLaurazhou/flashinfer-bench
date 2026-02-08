# CUresult cuGraphReleaseUserObject (CUgraph graph, CUuserObject object, unsigned int count)

Release a user object reference from a graph.

###### Parameters

**graph**

  - The graph that will release the reference
**object**

  - The user object to release a reference for


CUDA Driver API TRM-06703-001 _vRelease Version  |  491


Modules


**count**

  - The number of references to release, typically 1. Must be nonzero and not larger than INT_MAX.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Releases user object references owned by a graph.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.


See also:

cuUserObjectCreate, cuUserObjectRetain, cuUserObjectRelease, cuGraphRetainUserObject,
cuGraphCreate