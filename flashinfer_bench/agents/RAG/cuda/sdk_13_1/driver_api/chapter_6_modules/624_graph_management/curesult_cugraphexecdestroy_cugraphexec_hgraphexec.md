# CUresult cuGraphExecDestroy (CUgraphExec hGraphExec)

Destroys an executable graph.

###### Parameters

**hGraphExec**

  - Executable graph to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Destroys the executable graph specified by hGraphExec, as well as all of its executable nodes. If the
executable graph is in-flight, it will not be terminated, but rather freed asynchronously on completion.





See also:

cuGraphInstantiate, cuGraphUpload, cuGraphLaunch


CUDA Driver API TRM-06703-001 _vRelease Version  |  447


Modules