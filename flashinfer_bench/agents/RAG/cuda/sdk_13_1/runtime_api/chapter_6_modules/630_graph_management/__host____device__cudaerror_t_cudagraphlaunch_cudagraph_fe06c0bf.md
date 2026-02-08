# __host____device__cudaError_t cudaGraphLaunch (cudaGraphExec_t graphExec, cudaStream_t stream)

Launches an executable graph in a stream.

##### Parameters

**graphExec**

  - Executable graph to launch
**stream**

  - Stream in which to launch the graph

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Executes graphExec in stream. Only one instance of graphExec may be executing at a time.
Each launch is ordered behind both any previous work in stream and any previous launches of
graphExec. To execute a graph concurrently, it must be instantiated multiple times into multiple
executable graphs.

If any allocations created by graphExec remain unfreed (from a previous launch) and graphExec
was not instantiated with cudaGraphInstantiateFlagAutoFreeOnLaunch, the launch will fail with
cudaErrorInvalidValue.











See also:

cudaGraphInstantiate, cudaGraphUpload, cudaGraphExecDestroy


CUDA Runtime API vRelease Version  |  404


Modules