# enum CUlaunchAttributeID

Launch attributes enum; used as id field of CUlaunchAttribute

###### Values

**CU_LAUNCH_ATTRIBUTE_IGNORE = 0**
Ignored entry, for convenient composition
**CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1**
Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::accessPolicyWindow.
**CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2**
Valid for graph nodes, launches. See CUlaunchAttributeValue::cooperative.
**CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3**
Valid for streams. See CUlaunchAttributeValue::syncPolicy.
**CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4**
Valid for graph nodes, launches. See CUlaunchAttributeValue::clusterDim.
**CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5**


CUDA Driver API TRM-06703-001 _vRelease Version  |  58


Modules


Valid for graph nodes, launches. See CUlaunchAttributeValue::clusterSchedulingPolicyPreference.
**CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6**
Valid for launches. Setting CUlaunchAttributeValue::programmaticStreamSerializationAllowed to
non-0 signals that the kernel will use programmatic means to resolve its stream dependency, so that
the CUDA runtime should opportunistically allow the grid's execution to overlap with the previous
kernel in the stream, if that kernel requests the overlap. The dependent launches can choose to wait
on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent
PTX instructions).
**CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7**
Valid for launches. Set CUlaunchAttributeValue::programmaticEvent to record the event.
Event recorded through this launch attribute is guaranteed to only trigger after all block in the
associated kernel trigger the event. A block can trigger the event through PTX launchdep.release
or CUDA builtin function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also
be inserted at the beginning of each block's execution if triggerAtBlockStart is set to non-0.
The dependent launches can choose to wait on the dependency using the programmatic sync
(cudaGridDependencySynchronize() or equivalent PTX instructions). Note that dependents
(including the CPU thread calling cuEventSynchronize()) are not guaranteed to observe the release
precisely when it is released. For example, cuEventSynchronize() may only observe the event
trigger long after the associated kernel has completed. This recording type is primarily meant for
establishing programmatic dependency between device tasks. Note also this type of dependency
allows, but does not guarantee, concurrent execution of tasks. The event supplied must not be
an interprocess or interop event. The event must disable timing (i.e. must be created with the
CU_EVENT_DISABLE_TIMING flag set).
**CU_LAUNCH_ATTRIBUTE_PRIORITY = 8**
Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::priority.
**CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9**
Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::memSyncDomainMap.
**CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10**
Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::memSyncDomain.
**CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11**
Valid for graph nodes, launches. Set CUlaunchAttributeValue::preferredClusterDim to allow the
kernel launch to specify a preferred substitute cluster dimension. Blocks may be grouped according
to either the dimensions specified with this attribute (grouped into a "preferred substitute cluster"),
or the one specified with CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION attribute
(grouped into a "regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall
be an integer multiple greater than zero of the regular cluster dimensions. The device will attempt

  - on a best-effort basis - to group thread blocks into preferred clusters over grouping them into
regular clusters. When it deems necessary (primarily when the device temporarily runs out of
physical resources to launch the larger preferred clusters), the device may switch to launch the
regular clusters instead to attempt to utilize as much of the physical device resources as possible.
Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its
type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster
dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred


CUDA Driver API TRM-06703-001 _vRelease Version  |  59


Modules


cluster indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular
clusters (2,0,0) and (3,0,0) and groups their blocks. This attribute will only take effect when a
regular cluster dimension has been specified. The preferred substitute cluster dimension must be
an integer multiple greater than zero of the regular cluster dimension and must divide the grid. It
must also be no more than `maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`.
Otherwise it must be less than the maximum value the driver can support. Otherwise, setting this
attribute to a value physically unable to fit on any particular device is permitted.
**CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12**
Valid for launches. Set CUlaunchAttributeValue::launchCompletionEvent to record the
event. Nominally, the event is triggered once all blocks of the kernel have begun execution.
Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel
A, B may wait until A is complete. Alternatively, blocks of B may begin before all blocks
of A have begun, for example if B can claim execution resources unavailable to A (e.g.
they run on different GPUs) or if B is a higher priority than A. Exercise caution if such an
ordering inversion could lead to deadlock. A launch completion event is nominally similar
to a programmatic event with triggerAtBlockStart set except that it is not visible to
cudaGridDependencySynchronize() and can be used with compute capability less than
9.0. The event supplied must not be an interprocess or interop event. The event must disable timing
(i.e. must be created with the CU_EVENT_DISABLE_TIMING flag set).
**CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13**
Valid for graph nodes, launches. This attribute is graphs-only, and
passing it to a launch in a non-capturing stream will result in an error.
CUlaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can
only be set to 0 or 1. Setting the field to 1 indicates that the corresponding kernel
node should be device-updatable. On success, a handle will be returned via
CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be passed to the
various device-side update functions to update the node's kernel parameters from within another
kernel. For more information on the types of device updates that can be made, as well as the
relevant limitations thereof, see cudaGraphKernelNodeUpdatesApply. Nodes which are deviceupdatable have additional restrictions compared to regular kernel nodes. Firstly, device-updatable
nodes cannot be removed from their graph via cuGraphDestroyNode. Additionally, once optedin to this functionality, a node cannot opt out, and any attempt to set the deviceUpdatable attribute
to 0 will result in an error. Device-updatable kernel nodes also cannot have their attributes copied
to/from another kernel node via cuGraphKernelNodeCopyAttributes. Graphs containing one or
more device-updatable nodes also do not allow multiple instantiation, and neither the graph nor its
instantiated version can be passed to cuGraphExecUpdate. If a graph contains device-updatable
nodes and updates those nodes from the device from within the graph, the graph must be uploaded
with cuGraphUpload before it is launched. For such a graph, if host-side executable graph updates
are made to the device-updatable nodes, the graph must be uploaded before it is launched again.
**CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14**
Valid for launches. On devices where the L1 cache and shared memory use the same hardware
resources, setting CUlaunchAttributeValue::sharedMemCarveout to a percentage between
0-100 signals the CUDA driver to set the shared memory carveout preference, in percent


CUDA Driver API TRM-06703-001 _vRelease Version  |  60


Modules


of the total shared memory for that kernel launch. This attribute takes precedence over
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT. This is only a hint,
and the CUDA driver can choose a different configuration if required for the launch.
**CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING = 16**
Valid for streams, graph nodes, launches. This attribute is a hint to the CUDA runtime that the
launch should attempt to make the kernel maximize its NVLINK utilization. When possible
to honor this hint, CUDA will assume each block in the grid launch will carry out an even
amount of NVLINK traffic, and make a best-effort attempt to adjust the kernel launch based
on that assumption. This attribute is a hint only. CUDA makes no functional or performance
guarantee. Its applicability can be affected by many different factors, including driver version
(i.e. CUDA doesn't guarantee the performance characteristics will be maintained between driver
versions or a driver update could alter or regress previously observed perf characteristics.)
It also doesn't guarantee a successful result, i.e. applying the attribute may not improve the
performance of either the targeted kernel or the encapsulating application. Valid values for
CUlaunchAttributeValue::nvlinkUtilCentricScheduling are 0 (disabled) and 1 (enabled).