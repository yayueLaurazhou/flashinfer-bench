# __host__cudaError_t cudaGraphCreate (cudaGraph_t *pGraph, unsigned int flags)

Creates a graph.

##### Parameters

**pGraph**

  - Returns newly created graph
**flags**

  - Graph creation flags, must be 0

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

Creates an empty graph, which is returned via pGraph.











See also:

cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddHostNode, cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode,
cudaGraphInstantiate, cudaGraphDestroy, cudaGraphGetNodes, cudaGraphGetRootNodes,
cudaGraphGetEdges, cudaGraphClone


CUDA Runtime API vRelease Version  |  351


Modules