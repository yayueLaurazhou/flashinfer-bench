# __host__cudaError_t cudaGraphNodeFindInClone (cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)

Finds a cloned version of a node.

##### Parameters

**pNode**

  - Returns handle to the cloned node
**originalNode**

  - Handle to the original node
**clonedGraph**

  - Cloned graph to query

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

This function returns the node in clonedGraph corresponding to originalNode in the original
graph.

clonedGraph must have been cloned from originalGraph via cudaGraphClone.
originalNode must have been in originalGraph at the time of the call to cudaGraphClone, and
the corresponding cloned node in clonedGraph must not have been removed. The cloned node is
then returned via pClonedNode.











See also:

cudaGraphClone


CUDA Runtime API vRelease Version  |  414


Modules