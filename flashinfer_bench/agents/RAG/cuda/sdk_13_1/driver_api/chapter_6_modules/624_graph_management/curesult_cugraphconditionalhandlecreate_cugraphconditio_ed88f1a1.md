# CUresult cuGraphConditionalHandleCreate (CUgraphConditionalHandle *pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags)

Create a conditional handle.

###### Parameters

**pHandle_out**

  - Pointer used to return the handle to the caller.
**hGraph**

  - Graph which will contain the conditional node using this handle.
**ctx**

  - Context for the handle and associated conditional node.


CUDA Driver API TRM-06703-001 _vRelease Version  |  438


Modules


**defaultLaunchValue**

  - Optional initial value for the conditional variable. Applied at the beginning of each graph
execution if CU_GRAPH_COND_ASSIGN_DEFAULT is set in flags.
**flags**

 - Currently must be CU_GRAPH_COND_ASSIGN_DEFAULT or 0.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Creates a conditional handle associated with hGraph.

The conditional handle must be associated with a conditional node in this graph or one of its children.

Handles not associated with a conditional node may cause graph instantiation to fail.

Handles can only be set from the context with which they are associated.





See also:

cuGraphAddNode