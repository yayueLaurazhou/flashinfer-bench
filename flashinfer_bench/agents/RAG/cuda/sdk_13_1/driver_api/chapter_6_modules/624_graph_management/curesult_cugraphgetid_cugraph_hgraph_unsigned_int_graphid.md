# CUresult cuGraphGetId (CUgraph hGraph, unsigned int *graphId)

Returns the id of a given graph.

###### Parameters

**hGraph**

  - Graph to query
**graphId**

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the id of hGraph in *graphId. The value in *graphId will match that referenced by
cuGraphDebugDotPrint.


See also:

cuGraphGetNodes, cuGraphDebugDotPrint cuGraphNodeGetContainingGraph
cuGraphNodeGetLocalId cuGraphNodeGetToolsId cuGraphExecGetId