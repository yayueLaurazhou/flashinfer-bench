# CUresult cuGraphNodeSetEnabled (CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled)

Enables or disables the specified node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node


CUDA Driver API TRM-06703-001 _vRelease Version  |  489


Modules


**hNode**

  - Node from the graph from which graphExec was instantiated
**isEnabled**

  - Node is enabled if != 0, otherwise the node is disabled

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets hNode to be either enabled or disabled. Disabled nodes are functionally equivalent to empty
nodes until they are reenabled. Existing node parameters are not affected by disabling/enabling the
node.

The node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

If hNode is a device-updatable kernel node, the next upload/launch of hGraphExec will overwrite
any previous device-side updates. Additionally, applying host updates to a device-updatable kernel
node while it is being updated from the device will result in undefined behavior.


Note:


Currently only kernel, memset and memcpy nodes are supported.





See also:

cuGraphNodeGetEnabled, cuGraphExecUpdate, cuGraphInstantiate cuGraphLaunch


CUDA Driver API TRM-06703-001 _vRelease Version  |  490


Modules