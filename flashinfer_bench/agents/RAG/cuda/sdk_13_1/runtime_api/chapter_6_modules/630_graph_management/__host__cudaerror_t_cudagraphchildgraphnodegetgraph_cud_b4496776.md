# __host__cudaError_t cudaGraphChildGraphNodeGetGraph (cudaGraphNode_t node, cudaGraph_t *pGraph)

Gets a handle to the embedded graph of a child graph node.

##### Parameters

**node**

  - Node to get the embedded graph for
**pGraph**

  - Location to store a handle to the graph

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Gets a handle to the embedded graph in a child graph node. This call does not clone the graph. Changes
to the graph will be reflected in the node, and the node retains ownership of the graph.

Allocation and free nodes cannot be added to the returned graph. Attempting to do so will return an
error.


CUDA Runtime API vRelease Version  |  347


Modules











See also:

cudaGraphAddChildGraphNode, cudaGraphNodeFindInClone