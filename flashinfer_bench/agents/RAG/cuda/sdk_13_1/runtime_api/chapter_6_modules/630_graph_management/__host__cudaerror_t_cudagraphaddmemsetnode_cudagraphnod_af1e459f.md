# __host__cudaError_t cudaGraphAddMemsetNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaMemsetParams *pMemsetParams)

Creates a memset node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**pMemsetParams**

  - Parameters for the memory set

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Creates a new memset node and adds it to graph with numDependencies dependencies specified
via pDependencies. It is possible for numDependencies to be 0, in which case the node will be
placed at the root of the graph. pDependencies may not have any duplicate entries. A handle to the
new node will be returned in pGraphNode.

The element size must be 1, 2, or 4 bytes. When the graph is launched, the node will perform the
memset described by pMemsetParams.











CUDA Runtime API vRelease Version  |  345


Modules


See also:

cudaGraphAddNode, cudaMemset2D, cudaGraphMemsetNodeGetParams,
cudaGraphMemsetNodeSetParams, cudaGraphCreate, cudaGraphDestroyNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddHostNode, cudaGraphAddMemcpyNode