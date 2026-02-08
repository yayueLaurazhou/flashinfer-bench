# __device__cudaError_t cudaGraphKernelNodeSetGridDim (cudaGraphDeviceNode_t node, dim3 gridDim)

Updates the grid dimensions of the given kernel node.

##### Parameters

**node**

  - The node to update
**gridDim**

  - The grid dimensions to set

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the grid dimensions of node to gridDim. node must be device-updatable, and must reside upon
the same device as thecalling kernel.

If this function is called for the node's immediate dependent and that dependent is configured for
programmatic dependent launch, then a memory fence must be invoked via __threadfence() before
kickoff of the dependent is triggered via cudaTriggerProgrammaticLaunchCompletion() to ensure that
the update is visible to that dependent node before it is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphKernelNodeSetParam, cudaGraphKernelNodeSetEnabled,
cudaGraphKernelNodeUpdatesApply


CUDA Runtime API vRelease Version  |  399


Modules