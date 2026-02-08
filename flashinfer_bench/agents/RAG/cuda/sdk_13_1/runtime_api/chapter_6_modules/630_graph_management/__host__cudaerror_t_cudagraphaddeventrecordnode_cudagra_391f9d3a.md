# __host__cudaError_t cudaGraphAddEventRecordNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event)

Creates an event record node and adds it to a graph.

##### Parameters

**pGraphNode**
**graph**
**pDependencies**
**numDependencies**

  - Number of dependencies
**event**

  - Event for the node

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new event record node and adds it to hGraph with numDependencies dependencies
specified via dependencies and event specified in event. It is possible for numDependencies
to be 0, in which case the node will be placed at the root of the graph. dependencies may not have
any duplicate entries. A handle to the new node will be returned in phGraphNode.

Each launch of the graph will record event to capture execution of the node's dependencies.

These nodes may not be used in loops or conditionals.











CUDA Runtime API vRelease Version  |  327


Modules


See also:

cudaGraphAddNode, cudaGraphAddEventWaitNode, cudaEventRecordWithFlags,
cudaStreamWaitEvent, cudaGraphCreate, cudaGraphDestroyNode, cudaGraphAddChildGraphNode,
cudaGraphAddEmptyNode, cudaGraphAddKernelNode, cudaGraphAddMemcpyNode,
cudaGraphAddMemsetNode