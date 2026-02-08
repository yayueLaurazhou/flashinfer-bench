# __host__cudaError_t cudaGraphInstantiateWithParams (cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams *instantiateParams)

Creates an executable graph from a graph.

##### Parameters

**pGraphExec**

  - Returns instantiated graph
**graph**

  - Graph to instantiate
**instantiateParams**

  - Instantiation parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Instantiates graph as an executable graph according to the instantiateParams structure. The
graph is validated for any structural constraints or intra-node constraints which were not previously
validated. If instantiation is successful, a handle to the instantiated graph is returned in pGraphExec.

instantiateParams controls the behavior of instantiation and subsequent graph launches, as well
as returning more detailed information in the event of an error. cudaGraphInstantiateParams is defined
as:


The flags field controls the behavior of instantiation and subsequent graph launches. Valid flags are:

cudaGraphInstantiateFlagAutoFreeOnLaunch, which configures a graph containing memory

##### **‣**

allocation nodes to automatically free any unfreed memory allocations before the graph is
relaunched.

cudaGraphInstantiateFlagUpload, which will perform an upload of the graph into

##### **‣**

uploadStream once the graph has been instantiated.

cudaGraphInstantiateFlagDeviceLaunch, which configures the graph for launch from the device. If

##### **‣**

this flag is passed, the executable graph handle returned can be used to launch the graph from both
the host and device. This flag can only be used on platforms which support unified addressing. This
flag cannot be used in conjunction with cudaGraphInstantiateFlagAutoFreeOnLaunch.


CUDA Runtime API vRelease Version  |  393


Modules


cudaGraphInstantiateFlagUseNodePriority, which causes the graph to use the priorities from

##### **‣**

the per-node attributes rather than the priority of the launch stream during execution. Note that
priorities are only available on kernel nodes, and are copied from stream priority during stream
capture.

If graph contains any allocation or free nodes, there can be at most one executable graph in existence
for that graph at a time. An attempt to instantiate a second executable graph before destroying the
first with cudaGraphExecDestroy will result in an error. The same also applies if graph contains any
device-updatable kernel nodes.

If graph contains kernels which call device-side cudaGraphLaunch() from multiple devices, this will
result in an error.

Graphs instantiated for launch on the device have additional restrictions which do not apply to host
graphs:

The graph's nodes must reside on a single device.

##### **‣**

The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

##### **‣**

The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.

##### **‣**

Operation-specific restrictions are outlined below.
Kernel nodes:

##### **‣**

Use of CUDA Dynamic Parallelism is not permitted.

##### **‣**

Cooperative launches are permitted as long as MPS is not in use.

##### **‣**

Memcpy nodes:

##### **‣**

Only copies involving device memory and/or pinned device-mapped host memory are

##### **‣**

permitted.
Copies involving CUDA arrays are not permitted.

##### **‣**

Both operands must be accessible from the current device, and the current device must match

##### **‣**

the device of other nodes in the graph.

In the event of an error, the result_out and errNode_out fields will contain more information
about the nature of the error. Possible error reporting includes:

cudaGraphInstantiateError, if passed an invalid value or if an unexpected error occurred which is

##### **‣**

described by the return value of the function. errNode_out will be set to NULL.
cudaGraphInstantiateInvalidStructure, if the graph structure is invalid. will be set

##### ‣ errNode_out

to one of the offending nodes.
cudaGraphInstantiateNodeOperationNotSupported, if the graph is instantiated for device

##### **‣**

launch but contains a node of an unsupported node type, or a node which performs unsupported
operations, such as use of CUDA dynamic parallelism within a kernel node. errNode_out will
be set to this node.
cudaGraphInstantiateMultipleDevicesNotSupported, if the graph is instantiated for device

##### **‣**

launch but a node’s device differs from that of another node. This error can also be returned


CUDA Runtime API vRelease Version  |  394


Modules



if a graph is not instantiated for device launch and it contains kernels which call device-side
cudaGraphLaunch() from multiple devices. errNode_out will be set to this node.

If instantiation is successful, result_out will be set to cudaGraphInstantiateSuccess, and
hErrNode_out will be set to NULL.











See also:

cudaGraphCreate, cudaGraphInstantiate, cudaGraphInstantiateWithFlags, cudaGraphExecDestroy