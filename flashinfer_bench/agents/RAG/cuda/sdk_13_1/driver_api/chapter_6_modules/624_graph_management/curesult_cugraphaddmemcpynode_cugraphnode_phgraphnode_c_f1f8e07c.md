# CUresult cuGraphAddMemcpyNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx)

Creates a memcpy node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**copyParams**

  - Parameters for the memory copy
**ctx**

  - Context on which to run the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new memcpy node and adds it to hGraph with numDependencies dependencies
specified via dependencies. It is possible for numDependencies to be 0, in which case the node
will be placed at the root of the graph. dependencies may not have any duplicate entries. A handle
to the new node will be returned in phGraphNode.


CUDA Driver API TRM-06703-001 _vRelease Version  |  430


Modules


When the graph is launched, the node will perform the memcpy described by copyParams. See
cuMemcpy3D() for a description of the structure and its restrictions.

Memcpy nodes have some additional restrictions with regards to managed memory, if
the system contains at least one device which has a zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If one or more of the operands
refer to managed memory, then using the memory type CU_MEMORYTYPE_UNIFIED is disallowed
for those operand(s). The managed memory will be treated as residing on either the host or the device,
depending on which memory type is specified.





See also:

cuGraphAddNode, cuMemcpy3D, cuGraphMemcpyNodeGetParams,
cuGraphMemcpyNodeSetParams, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddHostNode, cuGraphAddMemsetNode