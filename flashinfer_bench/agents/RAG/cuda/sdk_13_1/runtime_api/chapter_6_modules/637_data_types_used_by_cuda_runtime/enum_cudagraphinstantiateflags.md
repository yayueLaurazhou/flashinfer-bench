# enum cudaGraphInstantiateFlags

Flags for instantiating a graph

##### Values

**cudaGraphInstantiateFlagAutoFreeOnLaunch = 1**
Automatically free memory allocated in a graph before relaunching.
**cudaGraphInstantiateFlagUpload = 2**
Automatically upload the graph after instantiation. Only supported by
cudaGraphInstantiateWithParams. The upload will be performed using the stream provided in
instantiateParams.
**cudaGraphInstantiateFlagDeviceLaunch = 4**
Instantiate the graph to be launchable from the device. This flag can only be used on
platforms which support unified addressing. This flag cannot be used in conjunction with
cudaGraphInstantiateFlagAutoFreeOnLaunch.
**cudaGraphInstantiateFlagUseNodePriority = 8**
Run the graph using the per-node priority attributes rather than the priority of the stream it is
launched into.