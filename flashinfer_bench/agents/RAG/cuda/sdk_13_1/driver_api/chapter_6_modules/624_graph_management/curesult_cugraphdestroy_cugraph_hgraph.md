# CUresult cuGraphDestroy (CUgraph hGraph)

Destroys a graph.

###### Parameters

**hGraph**

  - Graph to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Destroys the graph specified by hGraph, as well as all of its nodes.





See also:

cuGraphCreate