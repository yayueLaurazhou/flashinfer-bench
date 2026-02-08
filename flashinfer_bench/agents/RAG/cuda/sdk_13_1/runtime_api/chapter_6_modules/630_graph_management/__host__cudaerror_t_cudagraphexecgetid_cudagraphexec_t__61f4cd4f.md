# __host__cudaError_t cudaGraphExecGetId (cudaGraphExec_t hGraphExec, unsigned int *graphID)

Returns the id of a given graph exec.

##### Parameters

**hGraphExec**

  - Graph to query
**graphID**

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the id of hGraphExec in *graphId. The value in *graphId matches that referenced by
cudaGraphDebugDotPrint.


See also:

cudaGraphGetNodes, cudaGraphDebugDotPrint cudaGraphNodeGetContainingGraph
cudaGraphNodeGetLocalId cudaGraphNodeGetToolsId cudaGraphGetId