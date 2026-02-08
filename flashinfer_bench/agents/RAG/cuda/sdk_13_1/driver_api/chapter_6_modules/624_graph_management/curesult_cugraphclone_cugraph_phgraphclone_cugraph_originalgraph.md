# CUresult cuGraphClone (CUgraph *phGraphClone, CUgraph originalGraph)

Clones a graph.

###### Parameters

**phGraphClone**

  - Returns newly created cloned graph
**originalGraph**

  - Graph to clone


CUDA Driver API TRM-06703-001 _vRelease Version  |  437


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

This function creates a copy of originalGraph and returns it in phGraphClone. All parameters
are copied into the cloned graph. The original graph may be modified after this call without affecting
the clone.

Child graph nodes in the original graph are recursively copied into the clone.


Note:


: Cloning is not supported for graphs which contain memory allocation nodes, memory free nodes, or
conditional nodes.





See also:

cuGraphCreate, cuGraphNodeFindInClone