# __host__cudaError_t cudaGraphExecKernelNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams *pNodeParams)

Sets the parameters for a kernel node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - kernel node from the graph from which graphExec was instantiated


CUDA Runtime API vRelease Version  |  366


Modules


**pNodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets the parameters of a kernel node in an executable graph hGraphExec. The node is identified
by the corresponding node node in the non-executable graph, from which the executable graph was
instantiated.

node must not have been removed from the original graph. All nodeParams fields may change, but
the following restrictions apply to func updates:

The owning device of the function cannot change.

##### **‣**

A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a

##### **‣**

function which uses CDP
A node whose function originally did not make device-side update calls cannot be updated to a

##### **‣**

function which makes device-side update calls.
If was not instantiated for device launch, a node whose function originally did

##### ‣ hGraphExec

not use device-side cudaGraphLaunch() cannot be updated to a function which uses device-side
cudaGraphLaunch() unless the node resides on the same device as nodes which contained such
calls at instantiate-time. If no such calls were present at instantiation, these updates cannot be
performed at all.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.

If node is a device-updatable kernel node, the next upload/launch of hGraphExec will overwrite any
previous device-side updates. Additionally, applying host updates to a device-updatable kernel node
while it is being updated from the device will result in undefined behavior.











CUDA Runtime API vRelease Version  |  367


Modules





See also:

cudaGraphExecNodeSetParams, cudaGraphAddKernelNode,
cudaGraphKernelNodeSetParams, cudaGraphExecMemcpyNodeSetParams,
cudaGraphExecMemsetNodeSetParams, cudaGraphExecHostNodeSetParams,
cudaGraphExecChildGraphNodeSetParams, cudaGraphExecEventRecordNodeSetEvent,
cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate