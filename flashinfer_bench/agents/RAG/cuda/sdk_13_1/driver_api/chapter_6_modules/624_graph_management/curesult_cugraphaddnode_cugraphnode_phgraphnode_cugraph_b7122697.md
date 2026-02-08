# CUresult cuGraphAddNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, CUgraphNodeParams *nodeParams)

Adds a node of arbitrary type to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**dependencyData**

  - Optional edge data for the dependencies. If NULL, the data is assumed to be default (zeroed) for
all dependencies.
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Specification of the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Creates a new node in hGraph described by nodeParams with numDependencies dependencies
specified via dependencies. numDependencies may be 0. dependencies may be null if
numDependencies is 0. dependencies may not have any duplicate entries.

nodeParams is a tagged union. The node type should be specified in the type field, and typespecific parameters in the corresponding union member. All unused bytes - that is, reserved0


CUDA Driver API TRM-06703-001 _vRelease Version  |  434


Modules


and all bytes past the utilized union member - must be set to zero. It is recommended to use brace
initialization or memset to ensure all bytes are initialized.

Note that for some node types, nodeParams may contain "out parameters" which are modified during
the call, such as nodeParams->alloc.dptr.

A handle to the new node will be returned in phGraphNode.





See also:

cuGraphCreate, cuGraphNodeSetParams, cuGraphExecNodeSetParams