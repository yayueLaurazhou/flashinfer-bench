# CUresult cuGraphNodeFindInClone (CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)

Finds a cloned version of a node.

###### Parameters

**phNode**

  - Returns handle to the cloned node
**hOriginalNode**

  - Handle to the original node
**hClonedGraph**

  - Cloned graph to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

This function returns the node in hClonedGraph corresponding to hOriginalNode in the original
graph.

hClonedGraph must have been cloned from hOriginalGraph via cuGraphClone.
hOriginalNode must have been in hOriginalGraph at the time of the call to cuGraphClone,


CUDA Driver API TRM-06703-001 _vRelease Version  |  483


Modules


and the corresponding cloned node in hClonedGraph must not have been removed. The cloned node
is then returned via phClonedNode.





See also:

cuGraphClone