# __host__cudaError_t cudaGraphInstantiate (cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize)

Creates an executable graph from a graph.

##### Parameters

**pGraphExec**

  - Returns instantiated graph
**graph**

  - Graph to instantiate
**pErrorNode**

  - In case of an instantiation error, this may be modified to indicate a node contributing to the error
**pLogBuffer**

  - A character buffer to store diagnostic messages
**bufferSize**

  - Size of the log buffer in bytes

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  477


Modules

##### Description

Instantiates graph as an executable graph. The graph is validated for any structural constraints or
intra-node constraints which were not previously validated. If instantiation is successful, a handle to the
instantiated graph is returned in pGraphExec.

If there are any errors, diagnostic information may be returned in pErrorNode and pLogBuffer.
This is the primary way to inspect instantiation errors. The output will be null terminated unless the
diagnostics overflow the buffer. In this case, they will be truncated, and the last byte can be inspected
to determine if truncation occurred.











See also:

cudaGraphInstantiateWithFlags, cudaGraphCreate, cudaGraphUpload, cudaGraphLaunch,
cudaGraphExecDestroy