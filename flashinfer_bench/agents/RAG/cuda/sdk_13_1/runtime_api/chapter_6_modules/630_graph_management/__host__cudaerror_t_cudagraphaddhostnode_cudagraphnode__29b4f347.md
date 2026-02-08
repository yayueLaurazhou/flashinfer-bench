# __host__cudaError_t cudaGraphAddHostNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaHostNodeParams *pNodeParams)

Creates a host execution node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**pNodeParams**

  - Parameters for the host node

##### Returns

cudaSuccess, cudaErrorNotSupported, cudaErrorInvalidValue

##### Description

Creates a new CPU execution node and adds it to graph with numDependencies dependencies
specified via pDependencies and arguments specified in pNodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
pDependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

When the graph is launched, the node will invoke the specified CPU function. Host nodes are not
supported under MPS with pre-Volta GPUs.











CUDA Runtime API vRelease Version  |  332


Modules


See also:

cudaGraphAddNode, cudaLaunchHostFunc, cudaGraphHostNodeGetParams,
cudaGraphHostNodeSetParams, cudaGraphCreate, cudaGraphDestroyNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode