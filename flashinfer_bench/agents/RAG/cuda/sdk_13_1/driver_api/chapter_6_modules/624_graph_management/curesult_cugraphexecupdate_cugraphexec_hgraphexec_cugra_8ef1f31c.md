# CUresult cuGraphExecUpdate (CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo)

Check whether an executable graph can be updated with a graph and perform the update if possible.

###### Parameters

**hGraphExec**
The instantiated graph to be updated
**hGraph**
The graph containing the updated parameters
**resultInfo**
the error info structure

###### Returns

CUDA_SUCCESS, CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,

###### Description

Updates the node parameters in the instantiated graph specified by hGraphExec with the node
parameters in a topologically identical graph specified by hGraph.

Limitations:

Kernel nodes:

###### **‣**

The owning context of the function cannot change.

###### **‣**

A node whose function originally did not use CUDA dynamic parallelism cannot be updated to

###### **‣**

a function which uses CDP.
A node whose function originally did not make device-side update calls cannot be updated to a

###### **‣**

function which makes device-side update calls.
A cooperative node cannot be updated to a non-cooperative node, and vice-versa.

###### **‣**

If the graph was instantiated with

###### **‣**

CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, the priority attribute
cannot change. Equality is checked on the originally requested priority values, before they are
clamped to the device's supported range.
If was not instantiated for device launch, a node whose function originally did

###### ‣ hGraphExec

not use device-side cudaGraphLaunch() cannot be updated to a function which uses deviceside cudaGraphLaunch() unless the node resides on the same context as nodes which contained
such calls at instantiate-time. If no such calls were present at instantiation, these updates cannot
be performed at all.
Neither nor may contain device-updatable kernel nodes.

###### ‣ hGraph hGraphExec

CUDA Driver API TRM-06703-001 _vRelease Version  |  460


Modules


Memset and memcpy nodes:

###### **‣**

The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.

###### **‣**

The source/destination memory must be allocated from the same contexts as the original

###### **‣**

source/destination memory.
For 2d memsets, only address and assigned value may be updated.

###### **‣**

For 1d memsets, updating dimensions is also allowed, but may fail if the resulting operation

###### **‣**

doesn't map onto the work resources already allocated for the node.
Additional memcpy node restrictions:

###### **‣**

Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,

###### **‣**

CU_MEMORYTYPE_ARRAY, etc.) is not supported.
External semaphore wait nodes and record nodes:

###### **‣**

Changing the number of semaphores is not supported.

###### **‣**

Conditional nodes:

###### **‣**

Changing node parameters is not supported.

###### **‣**

Changing parameters of nodes within the conditional body graph is subject to the rules above.

###### **‣**

Conditional handle flags and default values are updated as part of the graph update.

###### **‣**

Note: The API may add further restrictions in future releases. The return code should always be
checked.

cuGraphExecUpdate sets the result member of resultInfo to
CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED under the following conditions:

The count of nodes directly in and differ, in which case resultInfo###### ‣ hGraphExec hGraph
>errorNode is set to NULL.
has more exit nodes than, in which case resultInfo->errorNode is set to one of

###### ‣ hGraph hGraph

the exit nodes in hGraph.
A node in has a different number of dependencies than the node from it is

###### ‣ hGraph hGraphExec

paired with, in which case resultInfo->errorNode is set to the node from hGraph.
A node in has a dependency that does not match with the corresponding dependency of

###### ‣ hGraph

the paired node from hGraphExec. resultInfo->errorNode will be set to the node from hGraph.
resultInfo->errorFromNode will be set to the mismatched dependency. The dependencies are
paired based on edge order and a dependency does not match when the nodes are already paired
based on other edges examined in the graph.

cuGraphExecUpdate sets the result member of resultInfo to:

CU_GRAPH_EXEC_UPDATE_ERROR if passed an invalid value.

###### **‣**

CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED if the graph topology

###### **‣**

changed
CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED if the type of a node

###### **‣**

changed, in which case hErrorNode_out is set to the node from hGraph.


CUDA Driver API TRM-06703-001 _vRelease Version  |  461


Modules


CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE if the

###### **‣**

function changed in an unsupported way(see note above), in which case hErrorNode_out is set
to the node from hGraph
CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED if any parameters to a

###### **‣**

node changed in a way that is not supported, in which case hErrorNode_out is set to the node
from hGraph.
CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED if any attributes of a node

###### **‣**

changed in a way that is not supported, in which case hErrorNode_out is set to the node from
hGraph.
CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED if something about a node is

###### **‣**

unsupported, like the node's type or configuration, in which case hErrorNode_out is set to the
node from hGraph

If the update fails for a reason not listed above, the result member of resultInfo will be set to
CU_GRAPH_EXEC_UPDATE_ERROR. If the update succeeds, the result member will be set to
CU_GRAPH_EXEC_UPDATE_SUCCESS.

cuGraphExecUpdate returns CUDA_SUCCESS when the updated was performed successfully. It
returns CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE if the graph update was not performed
because it included changes which violated constraints specific to instantiated graph update.





See also:

cuGraphInstantiate