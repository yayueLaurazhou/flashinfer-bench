# 6.2.8.7.5.3. Individual Node Update

###### 6.2.8.7.5.3. Individual Node Update[](#individual-node-update "Permalink to this headline")

Instantiated graph node parameters can be updated directly. This eliminates the overhead of instantiation as well as the overhead of creating a new `cudaGraph_t`. If the number of nodes requiring update is small relative to the total number of nodes in the graph, it is better to update the nodes individually. The following methods are available for updating `cudaGraphExec_t` nodes:

* `cudaGraphExecKernelNodeSetParams()`
* `cudaGraphExecMemcpyNodeSetParams()`
* `cudaGraphExecMemsetNodeSetParams()`
* `cudaGraphExecHostNodeSetParams()`
* `cudaGraphExecChildGraphNodeSetParams()`
* `cudaGraphExecEventRecordNodeSetEvent()`
* `cudaGraphExecEventWaitNodeSetEvent()`
* `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
* `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`

Please see the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.