# __host__cudaError_t cudaGraphExecGetFlags (cudaGraphExec_t graphExec, unsigned long long *flags)

Query the instantiation flags of an executable graph.

##### Parameters

**graphExec**

  - The executable graph to query
**flags**

  - Returns the instantiation flags

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the flags that were passed to instantiation for the given executable graph.
cudaGraphInstantiateFlagUpload will not be returned by this API as it does not affect the resulting
executable graph.











See also:

cudaGraphInstantiate, cudaGraphInstantiateWithFlags, cudaGraphInstantiateWithParams


CUDA Runtime API vRelease Version  |  364


Modules