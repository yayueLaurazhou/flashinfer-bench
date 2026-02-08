# __host__cudaError_t cudaGraphDestroyNode (cudaGraphNode_t node)

Remove a node from the graph.

##### Parameters

**node**

  - Node to remove

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Removes node from its graph. This operation also severs any dependencies of other nodes on node
and vice versa.

Dependencies cannot be removed from graphs which contain allocation or free nodes. Any attempt to
do so will return an error.











See also:


CUDA Runtime API vRelease Version  |  353


Modules


cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddHostNode, cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode