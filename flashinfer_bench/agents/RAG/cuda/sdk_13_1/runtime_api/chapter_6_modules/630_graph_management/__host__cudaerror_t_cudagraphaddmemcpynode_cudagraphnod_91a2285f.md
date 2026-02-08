# __host__cudaError_t cudaGraphAddMemcpyNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaMemcpy3DParms *pCopyParams)

Creates a memcpy node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**pCopyParams**

  - Parameters for the memory copy

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new memcpy node and adds it to graph with numDependencies dependencies specified
via pDependencies. It is possible for numDependencies to be 0, in which case the node will be
placed at the root of the graph. pDependencies may not have any duplicate entries. A handle to the
new node will be returned in pGraphNode.


CUDA Runtime API vRelease Version  |  337


Modules


When the graph is launched, the node will perform the memcpy described by pCopyParams. See
cudaMemcpy3D() for a description of the structure and its restrictions.

Memcpy nodes have some additional restrictions with regards to managed memory, if
the system contains at least one device which has a zero value for the device attribute
cudaDevAttrConcurrentManagedAccess.











See also:

cudaGraphAddNode, cudaMemcpy3D, cudaGraphAddMemcpyNodeToSymbol,
cudaGraphAddMemcpyNodeFromSymbol, cudaGraphAddMemcpyNode1D,
cudaGraphMemcpyNodeGetParams, cudaGraphMemcpyNodeSetParams, cudaGraphCreate,
cudaGraphDestroyNode, cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode,
cudaGraphAddKernelNode, cudaGraphAddHostNode, cudaGraphAddMemsetNode