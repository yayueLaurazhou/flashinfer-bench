# __host__cudaError_t cudaGraphConditionalHandleCreate_v2 (cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags)

Create a conditional handle.

##### Parameters

**pHandle_out**

  - Pointer used to return the handle to the caller.
**graph**

  - Graph which will contain the conditional node using this handle.
**ctx**

  - Execution context for the handle and associated conditional node. If NULL, current context will be
used.
**defaultLaunchValue**

  - Optional initial value for the conditional variable. Applied at the beginning of each graph
execution if cudaGraphCondAssignDefault is set in flags.
**flags**

  - Currently must be cudaGraphCondAssignDefault or 0.

##### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

##### Description

Creates a conditional handle associated with hGraph.

The conditional handle must be associated with a conditional node in this graph or one of its children.

Handles not associated with a conditional node may cause graph instantiation to fail.





CUDA Runtime API vRelease Version  |  350


Modules



See also:

cuGraphAddNode,