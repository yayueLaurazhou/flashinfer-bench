# __host__cudaError_t cudaGraphGetId (cudaGraph_t hGraph, unsigned int *graphID)

Returns the id of a given graph.

##### Parameters

**hGraph**

  - Graph to query
**graphID**

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  384


Modules

##### Description

Returns the id of hGraph in *graphId. The value in *graphId matches that referenced by
cudaGraphDebugDotPrint.


See also:

cudaGraphGetNodes, cudaGraphDebugDotPrint cudaGraphNodeGetContainingGraph
cudaGraphNodeGetLocalId cudaGraphNodeGetToolsId cudaGraphExecGetId