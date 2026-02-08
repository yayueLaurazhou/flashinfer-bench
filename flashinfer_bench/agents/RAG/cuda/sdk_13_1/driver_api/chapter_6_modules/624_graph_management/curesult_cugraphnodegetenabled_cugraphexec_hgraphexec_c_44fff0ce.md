# CUresult cuGraphNodeGetEnabled (CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled)

Query whether a node in the given graphExec is enabled.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Node from the graph from which graphExec was instantiated
**isEnabled**

  - Location to return the enabled status of the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets isEnabled to 1 if hNode is enabled, or 0 if hNode is disabled.

The node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.


Note:

**‣** Currently only kernel, memset and memcpy nodes are supported.

**‣** This function will not reflect device-side updates for device-updatable kernel nodes.





See also:

cuGraphNodeSetEnabled, cuGraphExecUpdate, cuGraphInstantiate cuGraphLaunch


CUDA Driver API TRM-06703-001 _vRelease Version  |  487


Modules