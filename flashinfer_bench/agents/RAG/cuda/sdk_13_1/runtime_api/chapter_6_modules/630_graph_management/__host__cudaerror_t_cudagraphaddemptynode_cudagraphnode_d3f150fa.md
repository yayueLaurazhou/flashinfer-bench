# __host__cudaError_t cudaGraphAddEmptyNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies)

Creates an empty node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new node which performs no operation, and adds it to graph with numDependencies
dependencies specified via pDependencies. It is possible for numDependencies to be 0, in
which case the node will be placed at the root of the graph. pDependencies may not have any
duplicate entries. A handle to the new node will be returned in pGraphNode.

An empty node performs no operation during execution, but can be used for transitive ordering. For
example, a phased execution graph with 2 groups of n nodes with a barrier between them can be
represented using an empty node and 2*n dependency edges, rather than no empty node and n^2
dependency edges.



See also:


CUDA Runtime API vRelease Version  |  326


Modules


cudaGraphAddNode, cudaGraphCreate, cudaGraphDestroyNode, cudaGraphAddChildGraphNode,
cudaGraphAddKernelNode, cudaGraphAddHostNode, cudaGraphAddMemcpyNode,
cudaGraphAddMemsetNode