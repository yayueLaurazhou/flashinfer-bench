# CUresult cuLaunchKernelEx (const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)

Launches a CUDA function CUfunction or a CUDA kernel CUkernel with launch-time configuration.

###### Parameters

**config**

  - Config to launch
**f**

  - Function CUfunction or Kernel CUkernel to launch
**kernelParams**

  - Array of pointers to kernel parameters
**extra**

  - Extra options

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
CUDA_ERROR_LAUNCH_TIMEOUT,
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, CUDA_ERROR_NOT_FOUND

###### Description

Invokes the function CUfunction or the kernel CUkernel f with the specified launch-time configuration
config.

The CUlaunchConfig structure is defined as:


CUDA Driver API TRM-06703-001 _vRelease Version  |  399


Modules


where:

CUlaunchConfig::gridDimX is the width of the grid in blocks.

###### **‣**

CUlaunchConfig::gridDimY is the height of the grid in blocks.

###### **‣**

CUlaunchConfig::gridDimZ is the depth of the grid in blocks.

###### **‣**

CUlaunchConfig::blockDimX is the X dimension of each thread block.

###### **‣**

CUlaunchConfig::blockDimX is the Y dimension of each thread block.

###### **‣**

CUlaunchConfig::blockDimZ is the Z dimension of each thread block.

###### **‣**

CUlaunchConfig::sharedMemBytes is the dynamic shared-memory size per thread block in bytes.

###### **‣**

CUlaunchConfig::hStream is the handle to the stream to perform the launch in. The CUDA context

###### **‣**

associated with this stream must match that associated with function f.
CUlaunchConfig::attrs is an array of CUlaunchConfig::numAttrs continguous CUlaunchAttribute

###### **‣**

elements. The value of this pointer is not considered if CUlaunchConfig::numAttrs is zero.
However, in that case, it is recommended to set the pointer to NULL.
CUlaunchConfig::numAttrs is the number of attributes populating the first

###### **‣**

CUlaunchConfig::numAttrs positions of the CUlaunchConfig::attrs array.

Launch-time configuration is specified by adding entries to CUlaunchConfig::attrs. Each entry is an
attribute ID and a corresponding attribute value.

The CUlaunchAttribute structure is defined as:


CUlaunchAttribute::id is a unique enum identifying the attribute.

###### **‣**

CUlaunchAttribute::value is a union that hold the attribute value.

###### **‣**

The CUlaunchAttributeID enum is defined as:


CUDA Driver API TRM-06703-001 _vRelease Version  |  400


Modules


and the corresponding CUlaunchAttributeValue union as :


Setting CU_LAUNCH_ATTRIBUTE_COOPERATIVE to a non-zero value causes the kernel launch
to be a cooperative launch, with exactly the same usage and semantics of cuLaunchCooperativeKernel.

Setting CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION to a nonzero values causes the kernel to use programmatic means to resolve its stream dependency -- enabling
the CUDA runtime to opportunistically allow the grid's execution to overlap with the previous kernel in
the stream, if that kernel requests the overlap.

CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT records an event along with the kernel
launch. Event recorded through this launch attribute is guaranteed to only trigger after all block in the
associated kernel trigger the event. A block can trigger the event through PTX launchdep.release or
CUDA builtin function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be inserted
at the beginning of each block's execution if triggerAtBlockStart is set to non-0. Note that dependents
(including the CPU thread calling cuEventSynchronize()) are not guaranteed to observe the release
precisely when it is released. For example, cuEventSynchronize() may only observe the event trigger
long after the associated kernel has completed. This recording type is primarily meant for establishing
programmatic dependency between device tasks. The event supplied must not be an interprocess or
interop event. The event must disable timing (i.e. created with CU_EVENT_DISABLE_TIMING flag
set).


CUDA Driver API TRM-06703-001 _vRelease Version  |  401


Modules


CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT records an event along with the
kernel launch. Nominally, the event is triggered once all blocks of the kernel have begun execution.
Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel A, B may
wait until A is complete. Alternatively, blocks of B may begin before all blocks of A have begun, for
example:

If B can claim execution resources unavailable to A, for example if they run on different GPUs.

###### **‣**

If B is a higher priority than A.

###### **‣**

Exercise caution if such an ordering inversion could lead to deadlock. The event supplied must not
be an interprocess or interop event. The event must disable timing (i.e. must be created with the
CU_EVENT_DISABLE_TIMING flag set).

Setting CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE to 1 on a captured
launch causes the resulting kernel node to be device-updatable. This attribute is specific to graphs, and
passing it to a launch in a non-capturing stream results in an error. Passing a value other than 0 or 1 is
not allowed.

On success, a handle will be returned via
CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be passed to the various
device-side update functions to update the node's kernel parameters from within another kernel. For
more information on the types of device updates that can be made, as well as the relevant limitations
thereof, see cudaGraphKernelNodeUpdatesApply.

Kernel nodes which are device-updatable have additional restrictions compared to regular kernel
nodes. Firstly, device-updatable nodes cannot be removed from their graph via cuGraphDestroyNode.
Additionally, once opted-in to this functionality, a node cannot opt out, and any attempt to set the
attribute to 0 will result in an error. Graphs containing one or more device-updatable node also do not
allow multiple instantiation.

CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION allows the kernel launch
to specify a preferred substitute cluster dimension. Blocks may be grouped according to either the
dimensions specified with this attribute (grouped into a "preferred substitute cluster"), or the one
specified with CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION attribute (grouped into a
"regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall be an integer multiple
greater than zero of the regular cluster dimensions. The device will attempt - on a best-effort basis to group thread blocks into preferred clusters over grouping them into regular clusters. When it deems
necessary (primarily when the device temporarily runs out of physical resources to launch the larger
preferred clusters), the device may switch to launch the regular clusters instead to attempt to utilize as
much of the physical device resources as possible.

Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its
type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster
dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred cluster
indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular clusters
(2,0,0) and (3,0,0) and groups their blocks.


CUDA Driver API TRM-06703-001 _vRelease Version  |  402


Modules


This attribute will only take effect when a regular cluster dimension has been specified. The
preferred substitute The preferred substitute cluster dimension must be an integer multiple greater
than zero of the regular cluster dimension and must divide the grid. It must also be no more than
`maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`. Otherwise it must be less than
the maximum value the driver can support. Otherwise, setting this attribute to a value physically unable
to fit on any particular device is permitted.

The effect of other attributes is consistent with their effect when set via persistent APIs.

See cuStreamSetAttribute for

CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW

###### **‣**

CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY

###### **‣**

See cuFuncSetAttribute for

CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION

###### **‣**

CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE

###### **‣**

Kernel parameters to f can be specified in the same ways that they can be using cuLaunchKernel.

Note that the API can also be used to launch context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to
launch the kernel on will either be taken from the specified stream CUlaunchConfig::hStream or the
current context in case of NULL stream.





See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncSetCacheConfig, cuFuncGetAttribute,
cudaLaunchKernel, cudaLaunchKernelEx, cuLibraryGetKernel, cuKernelSetCacheConfig,
cuKernelGetAttribute, cuKernelSetAttribute