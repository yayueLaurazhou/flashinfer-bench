# CUresult cuGraphExecGetId (CUgraphExec hGraphExec, unsigned int *graphId)

Returns the id of a given graph exec.

###### Parameters

**hGraphExec**

  - Graph to query
**graphId**

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the id of hGraphExec in *graphId. The value in *graphId will match that referenced by
cuGraphDebugDotPrint.


See also:

cuGraphGetNodes, cuGraphDebugDotPrint cuGraphNodeGetContainingGraph
cuGraphNodeGetLocalId cuGraphNodeGetToolsId cuGraphGetId