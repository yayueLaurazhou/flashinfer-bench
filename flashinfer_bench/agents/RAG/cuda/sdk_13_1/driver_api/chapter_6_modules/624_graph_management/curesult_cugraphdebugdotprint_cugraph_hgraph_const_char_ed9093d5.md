# CUresult cuGraphDebugDotPrint (CUgraph hGraph, const char *path, unsigned int flags)

Write a DOT file describing graph structure.

###### Parameters

**hGraph**

  - The graph to create a DOT file from
**path**

  - The path to write the DOT file to
**flags**

  - Flags from CUgraphDebugDot_flags for specifying which additional node information to write

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM

###### Description

Using the provided hGraph, write to path a DOT formatted description of the graph. By default this
includes the graph topology, node types, node id, kernel names and memcpy direction. flags can be
specified to write more detailed information about each node type such as parameter values, kernel
attributes, node and function handles.


CUDA Driver API TRM-06703-001 _vRelease Version  |  440


Modules