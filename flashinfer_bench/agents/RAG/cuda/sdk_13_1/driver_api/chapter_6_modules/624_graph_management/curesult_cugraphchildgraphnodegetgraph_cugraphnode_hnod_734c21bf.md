# CUresult cuGraphChildGraphNodeGetGraph (CUgraphNode hNode, CUgraph *phGraph)

Gets a handle to the embedded graph of a child graph node.

###### Parameters

**hNode**

  - Node to get the embedded graph for
**phGraph**

  - Location to store a handle to the graph

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE,

###### Description

Gets a handle to the embedded graph in a child graph node. This call does not clone the graph. Changes
to the graph will be reflected in the node, and the node retains ownership of the graph.

Allocation and free nodes cannot be added to the returned graph. Attempting to do so will return an
error.





See also:

cuGraphAddChildGraphNode, cuGraphNodeFindInClone