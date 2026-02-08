# __host__cudaError_t cudaGraphAddEventWaitNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event)

Creates an event wait node and adds it to a graph.

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

Creates a new event wait node and adds it to hGraph with numDependencies dependencies
specified via dependencies and event specified in event. It is possible for numDependencies
to be 0, in which case the node will be placed at the root of the graph. dependencies may not have
any duplicate entries. A handle to the new node will be returned in phGraphNode.

The graph node will wait for all work captured in event. See cuEventRecord() for details on what is
captured by an event. The synchronization will be performed efficiently on the device when applicable.
event may be from a different context or device than the launch stream.

These nodes may not be used in loops or conditionals.









CUDA Runtime API vRelease Version  |  328


Modules





See also:

cudaGraphAddNode, cudaGraphAddEventRecordNode, cudaEventRecordWithFlags,
cudaStreamWaitEvent, cudaGraphCreate, cudaGraphDestroyNode, cudaGraphAddChildGraphNode,
cudaGraphAddEmptyNode, cudaGraphAddKernelNode, cudaGraphAddMemcpyNode,
cudaGraphAddMemsetNode