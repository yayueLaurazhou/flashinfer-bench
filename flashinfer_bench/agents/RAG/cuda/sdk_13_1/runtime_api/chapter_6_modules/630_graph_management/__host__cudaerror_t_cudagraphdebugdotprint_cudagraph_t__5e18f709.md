# __host__cudaError_t cudaGraphDebugDotPrint (cudaGraph_t graph, const char *path, unsigned int flags)

Write a DOT file describing graph structure.

##### Parameters

**graph**

  - The graph to create a DOT file from
**path**

  - The path to write the DOT file to
**flags**

  - Flags from cudaGraphDebugDotFlags for specifying which additional node information to write

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorOperatingSystem

##### Description

Using the provided graph, write to path a DOT formatted description of the graph. By default this
includes the graph topology, node types, node id, kernel names and memcpy direction. flags can be
specified to write more detailed information about each node type such as parameter values, kernel
attributes, node and function handles.