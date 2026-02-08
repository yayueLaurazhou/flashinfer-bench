# CUresult cuGraphNodeGetContainingGraph (CUgraphNode hNode, CUgraph *phGraph)

Returns the graph that contains a given graph node.

###### Parameters

**hNode**

  - Node to query
**phGraph**

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the graph that contains hNode in *phGraph. If hNode is in a child graph, the child graph it
is in is returned.


See also:

cuGraphGetNodes, cuGraphDebugDotPrint cuGraphNodeGetLocalId cuGraphNodeGetToolsId
cuGraphGetId cuGraphExecGetId


CUDA Driver API TRM-06703-001 _vRelease Version  |  484


Modules