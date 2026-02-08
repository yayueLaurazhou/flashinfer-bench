# enum CUgraphInstantiate_flags

Flags for instantiating a graph

###### Values

**CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1**
Automatically free memory allocated in a graph before relaunching.
**CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2**
Automatically upload the graph after instantiation. Only supported by
cuGraphInstantiateWithParams. The upload will be performed using the stream provided in
instantiateParams.
**CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4**
Instantiate the graph to be launchable from the device. This flag can only be used on
platforms which support unified addressing. This flag cannot be used in conjunction with
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.
**CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8**
Run the graph using the per-node priority attributes rather than the priority of the stream it is
launched into.