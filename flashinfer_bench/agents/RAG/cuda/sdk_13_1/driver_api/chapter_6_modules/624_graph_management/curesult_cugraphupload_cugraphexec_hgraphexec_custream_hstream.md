# CUresult cuGraphUpload (CUgraphExec hGraphExec, CUstream hStream)

Uploads an executable graph in a stream.

###### Parameters

**hGraphExec**

  - Executable graph to upload
**hStream**

  - Stream in which to upload the graph

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Uploads hGraphExec to the device in hStream without executing it. Uploads of the same
hGraphExec will be serialized. Each upload is ordered behind both any previous work in hStream
and any previous launches of hGraphExec. Uses memory cached by stream to back the allocations
owned by hGraphExec.





See also:

cuGraphInstantiate, cuGraphLaunch, cuGraphExecDestroy


CUDA Driver API TRM-06703-001 _vRelease Version  |  494


Modules