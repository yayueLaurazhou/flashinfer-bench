# __host__cudaError_t cudaGraphClone (cudaGraph_t *pGraphClone, cudaGraph_t originalGraph)

Clones a graph.

##### Parameters

**pGraphClone**

  - Returns newly created cloned graph
**originalGraph**

  - Graph to clone

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

##### Description

This function creates a copy of originalGraph and returns it in pGraphClone. All parameters
are copied into the cloned graph. The original graph may be modified after this call without affecting
the clone.

Child graph nodes in the original graph are recursively copied into the clone.


Note:


: Cloning is not supported for graphs which contain memory allocation nodes, memory free nodes, or
conditional nodes.


Note:


CUDA Runtime API vRelease Version  |  348


Modules









See also:

cudaGraphCreate, cudaGraphNodeFindInClone