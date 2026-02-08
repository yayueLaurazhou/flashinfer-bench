# CUresult cuGraphCreate (CUgraph *phGraph, unsigned int flags)

Creates a graph.

###### Parameters

**phGraph**

  - Returns newly created graph
**flags**

  - Graph creation flags, must be 0

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY


CUDA Driver API TRM-06703-001 _vRelease Version  |  439


Modules

###### Description

Creates an empty graph, which is returned via phGraph.





See also:

cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddHostNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode, cuGraphInstantiate,
cuGraphDestroy, cuGraphGetNodes, cuGraphGetRootNodes, cuGraphGetEdges, cuGraphClone