# __host__cudaError_t cudaGraphExecUpdate (cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo *resultInfo)

Check whether an executable graph can be updated with a graph and perform the update if possible.

##### Parameters

**hGraphExec**
The instantiated graph to be updated
**hGraph**
The graph containing the updated parameters
**resultInfo**
the error info structure

##### Returns

cudaSuccess, cudaErrorGraphExecUpdateFailure,

##### Description

Updates the node parameters in the instantiated graph specified by hGraphExec with the node
parameters in a topologically identical graph specified by hGraph.

Limitations:

Kernel nodes:

##### **‣**

The owning context of the function cannot change.

##### **‣**

A node whose function originally did not use CUDA dynamic parallelism cannot be updated to

##### **‣**

a function which uses CDP.
A node whose function originally did not make device-side update calls cannot be updated to a

##### **‣**

function which makes device-side update calls.
A cooperative node cannot be updated to a non-cooperative node, and vice-versa.

##### **‣**

If the graph was instantiated with cudaGraphInstantiateFlagUseNodePriority, the priority

##### **‣**

attribute cannot change. Equality is checked on the originally requested priority values, before
they are clamped to the device's supported range.
If was not instantiated for device launch, a node whose function originally did

##### ‣ hGraphExec

not use device-side cudaGraphLaunch() cannot be updated to a function which uses device-side
cudaGraphLaunch() unless the node resides on the same device as nodes which contained such
calls at instantiate-time. If no such calls were present at instantiation, these updates cannot be
performed at all.
Neither nor may contain device-updatable kernel nodes.

##### ‣ hGraph hGraphExec

Memset and memcpy nodes:

##### **‣**

CUDA Runtime API vRelease Version  |  377


Modules


The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.

##### **‣**

The source/destination memory must be allocated from the same contexts as the original

##### **‣**

source/destination memory.
For 2d memsets, only address and assigned value may be updated.

##### **‣**

For 1d memsets, updating dimensions is also allowed, but may fail if the resulting operation

##### **‣**

doesn't map onto the work resources already allocated for the node.
Additional memcpy node restrictions:

##### **‣**

Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,

##### **‣**

CU_MEMORYTYPE_ARRAY, etc.) is not supported.
Conditional nodes:

##### **‣**

Changing node parameters is not supported.

##### **‣**

Changing parameters of nodes within the conditional body graph is subject to the rules above.

##### **‣**

Conditional handle flags and default values are updated as part of the graph update.

##### **‣**

Note: The API may add further restrictions in future releases. The return code should always be
checked.

cudaGraphExecUpdate sets the result member of resultInfo to
cudaGraphExecUpdateErrorTopologyChanged under the following conditions:

The count of nodes directly in and differ, in which case resultInfo##### ‣ hGraphExec hGraph
>errorNode is set to NULL.
has more exit nodes than, in which case resultInfo->errorNode is set to one of

##### ‣ hGraph hGraph

the exit nodes in hGraph.
A node in has a different number of dependencies than the node from it is

##### ‣ hGraph hGraphExec

paired with, in which case resultInfo->errorNode is set to the node from hGraph.
A node in has a dependency that does not match with the corresponding dependency of

##### ‣ hGraph

the paired node from hGraphExec. resultInfo->errorNode will be set to the node from hGraph.
resultInfo->errorFromNode will be set to the mismatched dependency. The dependencies are
paired based on edge order and a dependency does not match when the nodes are already paired
based on other edges examined in the graph.

cudaGraphExecUpdate sets the result member of resultInfo to:

cudaGraphExecUpdateError if passed an invalid value.

##### **‣**

cudaGraphExecUpdateErrorTopologyChanged if the graph topology changed

##### **‣**

cudaGraphExecUpdateErrorNodeTypeChanged if the type of a node changed, in which case

##### **‣**

hErrorNode_out is set to the node from hGraph.
cudaGraphExecUpdateErrorFunctionChanged if the function of a kernel node changed (CUDA

##### **‣**

driver < 11.2)


CUDA Runtime API vRelease Version  |  378


Modules


cudaGraphExecUpdateErrorUnsupportedFunctionChange if the func field of a kernel changed in

##### **‣**

an unsupported way(see note above), in which case hErrorNode_out is set to the node from
hGraph
cudaGraphExecUpdateErrorParametersChanged if any parameters to a node changed in a way that

##### **‣**

is not supported, in which case hErrorNode_out is set to the node from hGraph
cudaGraphExecUpdateErrorAttributesChanged if any attributes of a node changed in a way that is

##### **‣**

not supported, in which case hErrorNode_out is set to the node from hGraph
cudaGraphExecUpdateErrorNotSupported if something about a node is unsupported, like the

##### **‣**

node's type or configuration, in which case hErrorNode_out is set to the node from hGraph

If the update fails for a reason not listed above, the result member of resultInfo will be
set to cudaGraphExecUpdateError. If the update succeeds, the result member will be set to
cudaGraphExecUpdateSuccess.

cudaGraphExecUpdate returns cudaSuccess when the updated was performed successfully. It returns
cudaErrorGraphExecUpdateFailure if the graph update was not performed because it included changes
which violated constraints specific to instantiated graph update.











See also:

cudaGraphInstantiate