# CUresult cuGraphExecKernelNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams)

Sets the parameters for a kernel node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - kernel node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  454


Modules

###### Description

Sets the parameters of a kernel node in an executable graph hGraphExec. The node is identified by
the corresponding node hNode in the non-executable graph, from which the executable graph was
instantiated.

hNode must not have been removed from the original graph. All nodeParams fields may change,
but the following restrictions apply to func updates:

The owning context of the function cannot change.

###### **‣**

A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a

###### **‣**

function which uses CDP
A node whose function originally did not make device-side update calls cannot be updated to a

###### **‣**

function which makes device-side update calls.
If was not instantiated for device launch, a node whose function originally did

###### ‣ hGraphExec

not use device-side cudaGraphLaunch() cannot be updated to a function which uses device-side
cudaGraphLaunch() unless the node resides on the same context as nodes which contained such
calls at instantiate-time. If no such calls were present at instantiation, these updates cannot be
performed at all.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

If hNode is a device-updatable kernel node, the next upload/launch of hGraphExec will overwrite
any previous device-side updates. Additionally, applying host updates to a device-updatable kernel
node while it is being updated from the device will result in undefined behavior.





See also:

cuGraphExecNodeSetParams, cuGraphAddKernelNode, cuGraphKernelNodeSetParams,
cuGraphExecMemcpyNodeSetParams, cuGraphExecMemsetNodeSetParams,
cuGraphExecHostNodeSetParams, cuGraphExecChildGraphNodeSetParams,
cuGraphExecEventRecordNodeSetEvent, cuGraphExecEventWaitNodeSetEvent,
cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate


CUDA Driver API TRM-06703-001 _vRelease Version  |  455


Modules