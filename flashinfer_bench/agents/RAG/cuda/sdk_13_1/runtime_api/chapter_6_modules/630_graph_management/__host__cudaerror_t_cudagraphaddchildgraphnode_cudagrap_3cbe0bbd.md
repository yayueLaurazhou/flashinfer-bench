# __host__cudaError_t cudaGraphAddChildGraphNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph)

Creates a child graph node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**childGraph**

  - The graph to clone into this node


CUDA Runtime API vRelease Version  |  323


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new node which executes an embedded graph, and adds it to graph with
numDependencies dependencies specified via pDependencies. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
pDependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

If childGraph contains allocation nodes, free nodes, or conditional nodes, this call will return an
error.

The node executes an embedded child graph. The child graph is cloned in this call.











See also:

cudaGraphAddNode, cudaGraphChildGraphNodeGetGraph, cudaGraphCreate,
cudaGraphDestroyNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddHostNode, cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode,
cudaGraphClone