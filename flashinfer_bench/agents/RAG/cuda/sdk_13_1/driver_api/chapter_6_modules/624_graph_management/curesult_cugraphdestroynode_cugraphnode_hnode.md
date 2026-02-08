# CUresult cuGraphDestroyNode (CUgraphNode hNode)

Remove a node from the graph.

###### Parameters

**hNode**

  - Node to remove

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Removes hNode from its graph. This operation also severs any dependencies of other nodes on
hNode and vice versa.

Nodes which belong to a graph which contains allocation or free nodes cannot be destroyed. Any
attempt to do so will return an error.


CUDA Driver API TRM-06703-001 _vRelease Version  |  441


Modules





See also:

cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddHostNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode