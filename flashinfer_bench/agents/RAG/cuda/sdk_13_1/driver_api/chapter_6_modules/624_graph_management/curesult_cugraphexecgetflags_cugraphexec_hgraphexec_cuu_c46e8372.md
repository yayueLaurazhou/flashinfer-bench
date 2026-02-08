# CUresult cuGraphExecGetFlags (CUgraphExec hGraphExec, cuuint64_t *flags)

Query the instantiation flags of an executable graph.

###### Parameters

**hGraphExec**

  - The executable graph to query
**flags**

  - Returns the instantiation flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Returns the flags that were passed to instantiation for the given executable graph.
CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD will not be returned by this API as it does not
affect the resulting executable graph.





See also:

cuGraphInstantiate, cuGraphInstantiateWithParams


CUDA Driver API TRM-06703-001 _vRelease Version  |  452


Modules