# __host__cudaError_t cudaGraphInstantiate (cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags)

Creates an executable graph from a graph.

##### Parameters

**pGraphExec**

  - Returns instantiated graph
**graph**

  - Graph to instantiate
**flags**

  - Flags to control instantiation. See CUgraphInstantiate_flags.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Instantiates graph as an executable graph. The graph is validated for any structural constraints or
intra-node constraints which were not previously validated. If instantiation is successful, a handle to the
instantiated graph is returned in pGraphExec.

The flags parameter controls the behavior of instantiation and subsequent graph launches. Valid
flags are:

cudaGraphInstantiateFlagAutoFreeOnLaunch, which configures a graph containing memory

##### **‣**

allocation nodes to automatically free any unfreed memory allocations before the graph is
relaunched.

cudaGraphInstantiateFlagDeviceLaunch, which configures the graph for launch from the

##### **‣**

device. If this flag is passed, the executable graph handle returned can be used to launch
the graph from both the host and device. This flag cannot be used in conjunction with
cudaGraphInstantiateFlagAutoFreeOnLaunch.

cudaGraphInstantiateFlagUseNodePriority, which causes the graph to use the priorities from

##### **‣**

the per-node attributes rather than the priority of the launch stream during execution. Note that
priorities are only available on kernel nodes, and are copied from stream priority during stream
capture.

If graph contains any allocation or free nodes, there can be at most one executable graph in existence
for that graph at a time. An attempt to instantiate a second executable graph before destroying the
first with cudaGraphExecDestroy will result in an error. The same also applies if graph contains any
device-updatable kernel nodes.


CUDA Runtime API vRelease Version  |  389


Modules


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

If graph is not instantiated for launch on the device but contains kernels which call device-side
cudaGraphLaunch() from multiple devices, this will result in an error.











See also:

cudaGraphInstantiateWithFlags, cudaGraphCreate, cudaGraphUpload, cudaGraphLaunch,
cudaGraphExecDestroy


CUDA Runtime API vRelease Version  |  390


Modules