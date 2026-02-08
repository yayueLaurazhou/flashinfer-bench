# CUresult cuGraphLaunch (CUgraphExec hGraphExec, CUstream hStream)

Launches an executable graph in a stream.

###### Parameters

**hGraphExec**

  - Executable graph to launch
**hStream**

  - Stream in which to launch the graph

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Executes hGraphExec in hStream. Only one instance of hGraphExec may be executing at a
time. Each launch is ordered behind both any previous work in hStream and any previous launches
of hGraphExec. To execute a graph concurrently, it must be instantiated multiple times into multiple
executable graphs.

If any allocations created by hGraphExec remain unfreed (from
a previous launch) and hGraphExec was not instantiated with
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, the launch will fail with
CUDA_ERROR_INVALID_VALUE.


CUDA Driver API TRM-06703-001 _vRelease Version  |  478


Modules





See also:

cuGraphInstantiate, cuGraphUpload, cuGraphExecDestroy