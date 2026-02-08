# __host__cudaError_t cudaGraphAddNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, const cudaGraphEdgeData *dependencyData, size_t numDependencies, cudaGraphNodeParams *nodeParams)

Adds a node of arbitrary type to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**dependencyData**

  - Optional edge data for the dependencies. If NULL, the data is assumed to be default (zeroed) for
all dependencies.
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Specification of the node

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction, cudaErrorNotSupported

##### Description

Creates a new node in graph described by nodeParams with numDependencies dependencies
specified via pDependencies. numDependencies may be 0. pDependencies may be null if
numDependencies is 0. pDependencies may not have any duplicate entries.


CUDA Runtime API vRelease Version  |  346


Modules


nodeParams is a tagged union. The node type should be specified in the type field, and typespecific parameters in the corresponding union member. All unused bytes - that is, reserved0
and all bytes past the utilized union member - must be set to zero. It is recommended to use brace
initialization or memset to ensure all bytes are initialized.

Note that for some node types, nodeParams may contain "out parameters" which are modified during
the call, such as nodeParams->alloc.dptr.

A handle to the new node will be returned in phGraphNode.











See also:

cudaGraphCreate, cudaGraphNodeSetParams, cudaGraphExecNodeSetParams